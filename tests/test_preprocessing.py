"""Unit and integration tests for the preprocessing and feature engineering module."""

import importlib
import numpy as np
import pandas as pd
import pytest

from insurance_mlops.features.preprocessor import (
    build_preprocessor,
    fit_preprocessor,
    load_preprocessor,
    run_preprocessing,
    save_preprocessor,
    transform_data,
)

stage_04 = importlib.import_module("mlops_pipeline.04_validate_outputs")
validate_preprocessing_outputs = stage_04.validate_preprocessing_outputs


@pytest.fixture
def sample_data():
    """Generates synthetic train and test DataFrames matching insurance schema."""
    train = pd.DataFrame({
        "age": [20, 40, 60],
        "bmi": [20.0, 30.0, 40.0],
        "children": [0, 1, 2],
        "sex": ["male", "female", "male"],
        "smoker": ["no", "yes", "no"],
        "region": ["northeast", "southwest", "northwest"],
        "charges": [2000.0, 15000.0, 8000.0],
    })
    test = pd.DataFrame({
        "age": [25, 50],
        "bmi": [22.0, 32.0],
        "children": [1, 0],
        "sex": ["female", "male"],
        "smoker": ["no", "no"],
        "region": ["southeast", "northeast"],
        "charges": [3000.0, 9000.0],
    })
    return train, test


def test_build_preprocessor():
    """Tests unfitted preprocessor structure and configuration."""
    num_cols = ["age", "bmi", "children"]
    cat_cols = ["sex", "smoker", "region"]
    ct = build_preprocessor(num_cols, cat_cols, scaler_type="standard")

    assert ct is not None
    transformer_names = [name for name, _, _ in ct.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_fit_and_transform(sample_data):
    """Tests that fitting on train and transforming yields expected shapes and column names."""
    train_df, test_df = sample_data
    num_cols = ["age", "bmi", "children"]
    cat_cols = ["sex", "smoker", "region"]

    preprocessor = build_preprocessor(num_cols, cat_cols)
    preprocessor = fit_preprocessor(preprocessor, train_df[num_cols + cat_cols])

    train_trans = transform_data(preprocessor, train_df[num_cols + cat_cols])
    test_trans = transform_data(preprocessor, test_df[num_cols + cat_cols])

    assert train_trans.shape[0] == 3
    assert test_trans.shape[0] == 2
    assert train_trans.shape[1] == test_trans.shape[1]
    assert not train_trans.isnull().any().any()
    assert not test_trans.isnull().any().any()


def test_preprocessor_serialization(sample_data, tmp_path):
    """Tests saving and loading preprocessor artifact."""
    train_df, _ = sample_data
    num_cols = ["age", "bmi", "children"]
    cat_cols = ["sex", "smoker", "region"]

    preprocessor = build_preprocessor(num_cols, cat_cols)
    preprocessor = fit_preprocessor(preprocessor, train_df[num_cols + cat_cols])
    orig_trans = transform_data(preprocessor, train_df[num_cols + cat_cols])

    save_path = tmp_path / "preprocessor.joblib"
    save_preprocessor(preprocessor, save_path)
    assert save_path.exists()

    loaded_preproc = load_preprocessor(save_path)
    loaded_trans = transform_data(loaded_preproc, train_df[num_cols + cat_cols])

    pd.testing.assert_frame_equal(orig_trans, loaded_trans)


def test_data_leakage_prevention(sample_data):
    """Verifies that scaler statistics are derived STRICTLY from train data."""
    train_df, test_df = sample_data
    num_cols = ["age", "bmi", "children"]
    cat_cols = ["sex", "smoker", "region"]

    preprocessor = build_preprocessor(num_cols, cat_cols)
    preprocessor = fit_preprocessor(preprocessor, train_df[num_cols + cat_cols])

    scaler = preprocessor.named_transformers_["num"]
    # train 'age' is [20, 40, 60] -> mean is 40.0
    expected_train_age_mean = 40.0
    actual_mean = scaler.mean_[0]
    assert np.isclose(actual_mean, expected_train_age_mean), (
        f"Scaler mean ({actual_mean}) does not match train mean ({expected_train_age_mean})!"
    )


def test_handle_unknown_categories(sample_data):
    """Verifies that unseen categories in test data do not crash the pipeline."""
    train_df, _ = sample_data
    num_cols = ["age", "bmi", "children"]
    cat_cols = ["sex", "smoker", "region"]

    preprocessor = build_preprocessor(num_cols, cat_cols)
    preprocessor = fit_preprocessor(preprocessor, train_df[num_cols + cat_cols])

    # Unseen region 'midwest'
    unseen_test = pd.DataFrame({
        "age": [30],
        "bmi": [25.0],
        "children": [0],
        "sex": ["male"],
        "smoker": ["no"],
        "region": ["midwest"],  # Not present in train
    })

    # Should not raise ValueError because handle_unknown='ignore'
    trans = transform_data(preprocessor, unseen_test)
    assert trans.shape[0] == 1
    # Region columns should all be 0 for 'midwest'
    region_cols = [c for c in trans.columns if "region" in c]
    assert (trans[region_cols].values == 0.0).all()


def test_run_preprocessing_pipeline():
    """Tests full preprocessing pipeline execution on processed data."""
    results = run_preprocessing("config/config.yaml")

    assert results["num_features"] == 11
    assert results["X_train_shape"] == (1070, 11)
    assert results["X_test_shape"] == (268, 11)


def test_validate_preprocessing_outputs_function():
    """Tests the Stage 04 output validation check."""
    assert validate_preprocessing_outputs("config/config.yaml") is True
