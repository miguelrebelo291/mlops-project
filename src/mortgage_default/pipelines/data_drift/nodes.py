"""Nodes for the data_drift pipeline.

Compares the training feature distribution (reference: 2000-2002) against
each post-training year (2003-2008) using Evidently's DataDriftPreset.

Uses origination_data_cleaned (Standard + NSD, all years) as the data
source — not features_inference — so the comparison includes the full
loan population including ARM/interest-only products from the NSD,
which are the most relevant for detecting the pre-crisis drift.
"""
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from mortgage_default.pipelines.model_train.nodes import (
    apply_feature_transformers,
    drop_unused_columns,
    engineer_date_features,
)

logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES = [
    "first_time_homebuyer_flag",
    "occupancy_status",
    "channel",
    "prepayment_penalty_flag",
    "amortization_type",
    "property_state",
    "property_type",
    "loan_purpose",
    "interest_only_indicator",
]

TARGET_COL = "default"
ID_COL = "loan_sequence_number"
DATE_FEATURES = [
    "first_payment_date_year",
    "first_payment_date_month",
    "maturity_date_year",
    "maturity_date_month",
    "loan_term_months",
]


def _prepare_year_slice(
    df: pd.DataFrame,
    year: int,
    feature_transformers: dict,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Extract one year from the cleaned dataset, apply transformations,
    and return a float64 DataFrame aligned to feature_columns.
    Excludes date-derived features to avoid trivial drift from time passing.
    """
    slice_df = df[df["year"] == year].drop(columns=["year"]).copy()

    slice_df = drop_unused_columns(slice_df)
    slice_df = engineer_date_features(slice_df)
    slice_df = apply_feature_transformers(slice_df, feature_transformers)

    slice_df = slice_df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    slice_df = slice_df.drop(columns=[c for c in DATE_FEATURES if c in slice_df.columns])

    analysis_features = [c for c in feature_columns if c not in DATE_FEATURES]
    slice_df = slice_df.reindex(columns=analysis_features, fill_value=0)
    slice_df = slice_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float64")

    return slice_df


def _build_column_mapping(feature_columns: list[str]) -> ColumnMapping:
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_columns and c not in DATE_FEATURES]
    num_cols = [c for c in feature_columns if c not in cat_cols and c not in DATE_FEATURES]

    column_mapping = ColumnMapping()
    column_mapping.numerical_features = num_cols
    column_mapping.categorical_features = cat_cols
    return column_mapping


def _run_evidently(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    column_mapping: ColumnMapping,
) -> dict[str, Any]:
    """Run DataDriftPreset and return the report as dict."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )
    return report.as_dict()


def analyze_data_drift(
    origination_data_cleaned: pd.DataFrame,
    features_train: pd.DataFrame,
    feature_transformers: dict,
    train_metadata: dict,
    parameters: dict,
) -> tuple[pd.DataFrame, dict]:
    """Detect feature drift year by year (2003-2008) vs training (2000-2002).

    Uses origination_data_cleaned (Standard + NSD) so the comparison
    includes the full loan population — ARM, interest-only, and other
    non-standard products that drove pre-crisis risk.

    Date-derived features (first_payment_date_year, maturity_date_year,
    etc.) are excluded because they drift trivially as time passes, which
    would distort the narrative.

    Args:
        origination_data_cleaned: full cleaned dataset, all years, Standard
            + NSD.
        features_train: transformed training features (reference, 2000-2002),
            used as the Evidently reference distribution.
        feature_transformers: artifact from model_train (fitted on 2000-2002).
        train_metadata: metadata from model_train (feature_columns, etc.)
        parameters: data_drift parameter group.
    Returns:
        drift_report: DataFrame with per-feature drift scores per year.
        drift_metrics: dict with dataset-level drift summary per year.
    """
    feature_columns = train_metadata["feature_columns"]
    output_dir = Path(parameters.get("output_dir", "data/08_reporting"))
    output_dir.mkdir(parents=True, exist_ok=True)

    test_years = parameters.get("test_years", [2003, 2004, 2005, 2006, 2007, 2008])

    # Prepare reference — training features, drop date columns + target/ID
    reference = features_train.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    reference = reference.drop(columns=[c for c in DATE_FEATURES if c in reference.columns])
    analysis_features = [c for c in feature_columns if c not in DATE_FEATURES]
    reference = reference.reindex(columns=analysis_features, fill_value=0)
    reference = reference.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float64")

    column_mapping = _build_column_mapping(feature_columns)

    logger.info(
        "Reference: %d loans (2000-2002) | Analysing years: %s",
        len(reference), test_years,
    )

    all_rows = []
    yearly_summary = []

    for year in test_years:
        current = _prepare_year_slice(
            origination_data_cleaned, year, feature_transformers, feature_columns
        )

        if current.empty:
            logger.warning("No data found for year %d — skipping", year)
            continue

        logger.info("Running drift for %d (%d loans)...", year, len(current))

        report_dict = _run_evidently(reference, current, column_mapping)

        dataset_result = report_dict["metrics"][0]["result"]
        drift_by_columns = report_dict["metrics"][1]["result"]["drift_by_columns"]

        for feature, details in drift_by_columns.items():
            all_rows.append({
                "year": year,
                "feature": feature,
                "drift_detected": bool(details["drift_detected"]),
                "drift_score": float(details["drift_score"]),
                "stattest_name": details.get("stattest_name", "N/A"),
            })

        n_drifted = int(dataset_result["number_of_drifted_columns"])
        n_total = int(dataset_result["number_of_columns"])

        yearly_summary.append({
            "year": year,
            "n_loans": len(current),
            "dataset_drift_detected": bool(dataset_result["dataset_drift"]),
            "drift_share": float(dataset_result["drift_share"]),
            "n_drifted_features": n_drifted,
            "n_features": n_total,
        })

        logger.info(
            "%d: %d/%d features drifted (%.0f%%) — dataset_drift=%s",
            year, n_drifted, n_total,
            dataset_result["drift_share"] * 100,
            dataset_result["dataset_drift"],
        )

    # Save HTML for the most recent year (2007) as the main report
    current_2007 = _prepare_year_slice(
        origination_data_cleaned, 2007, feature_transformers, feature_columns
    )
    full_report = Report(metrics=[DataDriftPreset()])
    full_report.run(
        reference_data=reference,
        current_data=current_2007,
        column_mapping=column_mapping,
    )
    html_path = output_dir / "data_drift_report.html"
    full_report.save_html(str(html_path))
    logger.info("Evidently HTML report (2007) saved to %s", html_path)

    drift_report = pd.DataFrame(all_rows)
    metrics = {
        "reference_label": parameters.get("reference_label", "train_2000_2002"),
        "n_reference_rows": int(len(reference)),
        "test_years": test_years,
        "yearly_summary": yearly_summary,
        "html_report_path": str(html_path),
        "features_excluded_from_drift": DATE_FEATURES,
    }

    return drift_report, metrics
