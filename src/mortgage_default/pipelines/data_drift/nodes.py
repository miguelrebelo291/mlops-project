from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPSILON = 1e-6


def _safe_distribution(values: pd.Series) -> pd.Series:
    """Return normalized value counts with missing values included."""
    values = values.astype("string").fillna("__MISSING__")
    return values.value_counts(normalize=True)


def _psi_from_distributions(reference_pct: np.ndarray, current_pct: np.ndarray) -> float:
    """Population Stability Index."""
    reference_pct = np.where(reference_pct <= 0, EPSILON, reference_pct)
    current_pct = np.where(current_pct <= 0, EPSILON, current_pct)

    return float(
        np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct))
    )


def _js_distance_from_distributions(reference_pct: np.ndarray, current_pct: np.ndarray) -> float:
    """Jensen-Shannon distance using base-2 logarithms."""

    reference_pct = np.where(reference_pct <= 0, EPSILON, reference_pct)
    current_pct = np.where(current_pct <= 0, EPSILON, current_pct)

    reference_pct = reference_pct / reference_pct.sum()
    current_pct = current_pct / current_pct.sum()

    midpoint = 0.5 * (reference_pct + current_pct)

    kl_ref = np.sum(reference_pct * np.log2(reference_pct / midpoint))
    kl_cur = np.sum(current_pct * np.log2(current_pct / midpoint))

    js_divergence = 0.5 * (kl_ref + kl_cur)

    return float(np.sqrt(js_divergence))


def _numeric_drift(
    reference: pd.Series,
    current: pd.Series,
    bins: int,
) -> tuple[float, float]:
    """Calculate PSI and JS distance for a numeric feature using reference quantile bins."""

    reference_numeric = pd.to_numeric(reference, errors="coerce")
    current_numeric = pd.to_numeric(current, errors="coerce")

    reference_numeric = reference_numeric.replace([np.inf, -np.inf], np.nan).dropna()
    current_numeric = current_numeric.replace([np.inf, -np.inf], np.nan).dropna()

    if reference_numeric.empty or current_numeric.empty:
        return 0.0, 0.0

    breakpoints = np.nanquantile(
        reference_numeric,
        np.linspace(0, 1, bins + 1),
    )
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) <= 2:
        return _categorical_drift(reference, current)

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    reference_counts, _ = np.histogram(reference_numeric, bins=breakpoints)
    current_counts, _ = np.histogram(current_numeric, bins=breakpoints)

    reference_pct = reference_counts / max(reference_counts.sum(), 1)
    current_pct = current_counts / max(current_counts.sum(), 1)

    psi = _psi_from_distributions(reference_pct, current_pct)
    js_distance = _js_distance_from_distributions(reference_pct, current_pct)

    return psi, js_distance


def _categorical_drift(
    reference: pd.Series,
    current: pd.Series,
) -> tuple[float, float]:
    """Calculate PSI and JS distance for categorical or low-cardinality features."""

    reference_dist = _safe_distribution(reference)
    current_dist = _safe_distribution(current)

    categories = sorted(set(reference_dist.index).union(set(current_dist.index)))

    reference_pct = np.array([reference_dist.get(category, 0.0) for category in categories])
    current_pct = np.array([current_dist.get(category, 0.0) for category in categories])

    psi = _psi_from_distributions(reference_pct, current_pct)
    js_distance = _js_distance_from_distributions(reference_pct, current_pct)

    return psi, js_distance


def _plot_psi_report(
    drift_report: pd.DataFrame,
    output_path: Path,
    threshold: float,
    top_n: int,
) -> None:
    """Create professor-style horizontal PSI bar plot with threshold line."""

    plot_data = drift_report.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(plot_data["feature"], plot_data["psi"])
    ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")

    ax.set_xlabel("PSI")
    ax.set_ylabel("Feature")
    ax.set_title("Top Feature Drift by PSI")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_data_drift(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, dict]:
    """Compare reference training features against current inference features."""

    target_col = parameters.get("target_column", "default")
    id_col = parameters.get("id_column", "loan_sequence_number")
    bins = int(parameters.get("bins", 10))
    drift_threshold = float(parameters.get("drift_threshold", 0.20))
    top_n_plot = int(parameters.get("top_n_plot", 20))
    max_unique_as_categorical = int(parameters.get("max_unique_as_categorical", 20))
    output_dir = Path(parameters.get("output_dir", "data/08_reporting"))

    reference_label = parameters.get("reference_label", "reference")
    current_label = parameters.get("current_label", "current")

    output_dir.mkdir(parents=True, exist_ok=True)

    reference = reference_features.copy()
    current = current_features.copy()

    reference = reference.drop(columns=[target_col, id_col], errors="ignore")
    current = current.drop(columns=[target_col, id_col], errors="ignore")

    common_features = sorted(set(reference.columns).intersection(set(current.columns)))

    rows = []

    for feature in common_features:
        reference_col = reference[feature]
        current_col = current[feature]

        reference_missing_rate = float(reference_col.isna().mean())
        current_missing_rate = float(current_col.isna().mean())

        reference_numeric = pd.to_numeric(reference_col, errors="coerce")
        current_numeric = pd.to_numeric(current_col, errors="coerce")

        numeric_ratio_reference = reference_numeric.notna().mean()
        numeric_ratio_current = current_numeric.notna().mean()

        n_unique = max(
            reference_col.nunique(dropna=True),
            current_col.nunique(dropna=True),
        )

        is_numeric_feature = (
            numeric_ratio_reference > 0.95
            and numeric_ratio_current > 0.95
            and n_unique > max_unique_as_categorical
        )

        if is_numeric_feature:
            feature_type = "numeric"
            psi, js_distance = _numeric_drift(reference_col, current_col, bins=bins)

            reference_mean = float(reference_numeric.mean())
            current_mean = float(current_numeric.mean())

        else:
            feature_type = "categorical"
            psi, js_distance = _categorical_drift(reference_col, current_col)

            reference_mean = None
            current_mean = None

        rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "psi": float(psi),
                "js_distance": float(js_distance),
                "reference_missing_rate": reference_missing_rate,
                "current_missing_rate": current_missing_rate,
                "missing_rate_difference": current_missing_rate - reference_missing_rate,
                "reference_mean": reference_mean,
                "current_mean": current_mean,
                "drift_detected": bool(psi >= drift_threshold),
            }
        )

    drift_report = pd.DataFrame(rows)

    if drift_report.empty:
        metrics = {
            "reference_label": reference_label,
            "current_label": current_label,
            "n_reference_rows": int(len(reference_features)),
            "n_current_rows": int(len(current_features)),
            "n_features_compared": 0,
            "drift_threshold": drift_threshold,
            "n_drifted_features": 0,
            "share_drifted_features": 0.0,
            "top_drifted_features": [],
            "psi_plot_path": None,
        }
        return drift_report, metrics

    drift_report = drift_report.sort_values("psi", ascending=False).reset_index(drop=True)

    plot_path = output_dir / "data_drift_psi_plot.png"
    _plot_psi_report(
        drift_report=drift_report,
        output_path=plot_path,
        threshold=drift_threshold,
        top_n=top_n_plot,
    )

    n_drifted = int(drift_report["drift_detected"].sum())

    metrics = {
        "reference_label": reference_label,
        "current_label": current_label,
        "n_reference_rows": int(len(reference_features)),
        "n_current_rows": int(len(current_features)),
        "n_features_compared": int(len(common_features)),
        "drift_threshold": drift_threshold,
        "n_drifted_features": n_drifted,
        "share_drifted_features": float(n_drifted / len(drift_report)),
        "top_drifted_features": drift_report.head(10)[
            ["feature", "feature_type", "psi", "js_distance", "drift_detected"]
        ].to_dict(orient="records"),
        "psi_plot_path": str(plot_path),
    }

    return drift_report, metrics