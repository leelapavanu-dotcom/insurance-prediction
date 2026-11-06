"""Unit and integration tests for data loading and Pydantic validation."""

import pytest
import pandas as pd
from pydantic import ValidationError

from insurance_mlops.data.loader import load_raw_data, split_data, run_data_pipeline
from insurance_mlops.data.validator import InsuranceRecord, validate_schema


def test_insurance_record_valid():
    """Tests that a valid record passes Pydantic schema validation."""
    record = InsuranceRecord(
        age=30,
        sex="female",
        bmi=24.5,
        children=2,
        smoker="no",
        region="southwest",
        charges=4500.0,
    )
    assert record.age == 30
    assert record.sex == "female"
    assert record.charges == 4500.0


def test_insurance_record_invalid_age():
    """Tests that an out-of-range age fails validation."""
    with pytest.raises(ValidationError):
        InsuranceRecord(
            age=10,  # Below 18 minimum
            sex="male",
            bmi=25.0,
            children=0,
            smoker="no",
            region="southeast",
        )


def test_insurance_record_invalid_categorical():
    """Tests that unrecognized categorical values fail validation."""
    with pytest.raises(ValidationError):
        InsuranceRecord(
            age=40,
            sex="other",  # Only 'female' or 'male' allowed
            bmi=25.0,
            children=1,
            smoker="no",
            region="southeast",
        )

    with pytest.raises(ValidationError):
        InsuranceRecord(
            age=40,
            sex="female",
            bmi=25.0,
            children=1,
            smoker="occasionally",  # Only 'yes' or 'no' allowed
            region="southeast",
        )


def test_validate_schema_missing_columns():
    """Tests that missing required columns raises ValueError."""
    bad_df = pd.DataFrame({"age": [25], "sex": ["male"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(bad_df, is_training=True)


def test_validate_schema_missing_values():
    """Tests that NaN/null values fail validation."""
    bad_df = pd.DataFrame({
        "age": [25, None],
        "sex": ["male", "female"],
        "bmi": [25.0, 30.0],
        "children": [0, 1],
        "smoker": ["no", "yes"],
        "region": ["northwest", "southeast"],
        "charges": [1000.0, 2000.0],
    })
    with pytest.raises(ValueError, match="Found .* missing values"):
        validate_schema(bad_df, is_training=True)


def test_validate_schema_duplicate_reporting_and_handling():
    """Tests that duplicate rows are detected and handled per configuration."""
    sample_df = pd.DataFrame({
        "age": [20, 20],
        "sex": ["male", "male"],
        "bmi": [25.0, 25.0],
        "children": [0, 0],
        "smoker": ["no", "no"],
        "region": ["southwest", "southwest"],
        "charges": [1500.0, 1500.0],
    })

    # Test retention (drop_duplicates=False)
    retained_df, report_retained = validate_schema(sample_df, is_training=True, drop_duplicates=False)
    assert len(retained_df) == 2
    assert report_retained.duplicate_count == 1
    assert report_retained.dropped_duplicates == 0

    # Test dropping (drop_duplicates=True)
    dropped_df, report_dropped = validate_schema(sample_df, is_training=True, drop_duplicates=True)
    assert len(dropped_df) == 1
    assert report_dropped.duplicate_count == 1
    assert report_dropped.dropped_duplicates == 1


def test_load_raw_data():
    """Tests raw dataset loading from default raw path."""
    df = load_raw_data()
    assert df.shape == (1338, 7)
    assert "charges" in df.columns


def test_split_data():
    """Tests train/test partition proportions and reproducibility."""
    df = load_raw_data()
    train_df, test_df = split_data(df, test_size=0.2, random_state=42)

    assert len(train_df) + len(test_df) == len(df)
    assert len(test_df) == int(len(df) * 0.2) or len(test_df) == round(len(df) * 0.2)
    assert "charges" in train_df.columns
    assert "charges" in test_df.columns


def test_run_data_pipeline(tmp_path):
    """Tests complete data ingestion and validation pipeline execution."""
    train_df, test_df, report = run_data_pipeline("config/config.yaml")

    assert report.is_valid is True
    assert report.duplicate_count == 1  # Standard 1 duplicate row in raw dataset
    assert len(train_df) > 1000
    assert len(test_df) > 200
