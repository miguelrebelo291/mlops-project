"""
This is a boilerplate pipeline 'data_quality'
generated using Kedro 1.3.1
"""
import logging

import great_expectations as gx
import pandas as pd

logger = logging.getLogger(__name__)

CRITICAL_EXPECTATIONS = [
    "expect_table_column_count_to_equal",
    "expect_column_to_exist",
    "expect_column_values_to_be_unique",
    "expect_column_values_to_be_in_set",  # default column
]


def run_data_quality(model_input_data: pd.DataFrame) -> pd.DataFrame:
    """Run data quality checks on the model input data using Great Expectations.

    Critical expectations (structure and target variable) will raise an error
    and stop the pipeline if they fail. Non-critical expectations log a warning.

    Args:
        model_input_data: joined origination and target dataset (all years combined).
    Returns:
        The same DataFrame if all critical checks pass.
    Raises:
        ValueError: if any critical expectation fails.
    """
    context = gx.get_context(mode="ephemeral")

    #Data Source
    data_source = context.data_sources.add_pandas(name="mortgage_datasource")
    data_asset = data_source.add_dataframe_asset(name="model_input_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch")

    # Suite 
    suite = context.suites.add(gx.ExpectationSuite(name="mortgage_quality_suite"))

    # Structure
    # 34 columns = 32 origination + 'default' + 'year'
    suite.add_expectation(gx.expectations.ExpectTableColumnCountToEqual(value=34))
    # 9 anos * ~50k empréstimos por ano
    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=300000, max_value=600000))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="loan_sequence_number"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="default"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="year"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="loan_sequence_number"))

    # Target
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="default", value_set=[0, 1]
    ))

    # Year range (2000-2008)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="year", min_value=2000, max_value=2008
    ))

    # credit_score
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="credit_score",
        value_set=list(range(300, 851)) + [9999]
    ))

    # original_ltv
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="original_ltv", min_value=6, max_value=999
    ))

    # original_interest_rate
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="original_interest_rate", min_value=0, max_value=30, strict_min=True
    ))

    # msa null rate
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(
        column="msa", mostly=0.80
    ))

    # Relationships
    suite.add_expectation(gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
        column_A="maturity_date",
        column_B="first_payment_date",
    ))
    suite.add_expectation(gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
        column_A="original_cltv",
        column_B="original_ltv",
        or_equal=True,
    ))

    # Validation definition
    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="mortgage_validation",
            data=batch_definition,
            suite=suite,
        )
    )

    results = validation_definition.run(batch_parameters={"dataframe": model_input_data})

    # Results
    critical_failures = []
    for result in results.results:
        expectation_type = result.expectation_config.type
        if result.success:
            logger.info("PASSED: %s", expectation_type)
        else:
            if expectation_type in CRITICAL_EXPECTATIONS:
                critical_failures.append(expectation_type)
                logger.error("CRITICAL FAILURE: %s", expectation_type)
            else:
                logger.warning("WARNING: %s", expectation_type)

    if critical_failures:
        raise ValueError(
            f"Data quality critical failures: {critical_failures}"
        )

    logger.info("Data quality checks passed successfully.")
    return model_input_data
