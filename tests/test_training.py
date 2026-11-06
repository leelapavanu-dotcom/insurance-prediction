"""Unit and integration tests for model training, comparison, and MLflow tracking."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import joblib
import numpy as np
import pandas as pd
import pytest

from insurance_mlops.models.trainer import (
    calculate_metrics,
    cross_validate_model,
    get_model_candidates,
    run_training_pipeline,
    save_training_outputs,
    train_and_compare_models,
)


@pytest.fixture
def synthetic_train_data():
    """Generates synthetic preprocessed features and targets for fast testing."""
    np.random.seed(42)
    n_samples = 60
    n_features = 11

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Target linearly dependent on first feature + noise
    y = pd.Series(X["feat_0"] * 1000.0 + 5000.0 + np.random.randn(n_samples) * 100.0, name="charges")
    return X, y


def test_calculate_metrics():
    """Tests metric calculation accuracy (MAE, RMSE, R2)."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    metrics = calculate_metrics(y_true, y_pred)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert np.isclose(metrics["mae"], (10 + 10 + 0) / 3.0, atol=1e-2)
    assert np.isclose(metrics["rmse"], np.sqrt((100 + 100 + 0) / 3.0), atol=1e-2)
    assert metrics["r2"] > 0.95


def test_get_model_candidates():
    """Tests that candidate model definitions are present and reproducible."""
    candidates = get_model_candidates(random_seed=42)

    assert "linear_regression" in candidates
    assert "random_forest" in candidates
    assert "gradient_boosting" in candidates
    assert len(candidates) == 3


def test_cross_validate_model(synthetic_train_data):
    """Tests cross-validation execution strictly on training data."""
    X, y = synthetic_train_data
    candidates = get_model_candidates(random_seed=42)
    lr = candidates["linear_regression"]

    cv_results = cross_validate_model(lr, X, y.to_numpy(), cv_folds=3, random_seed=42)

    for key in ["cv_rmse", "cv_rmse_std", "cv_mae", "cv_mae_std", "cv_r2", "cv_r2_std"]:
        assert key in cv_results
        assert isinstance(cv_results[key], float)
    assert cv_results["cv_rmse"] > 0.0


def test_train_and_compare_models_selection(synthetic_train_data):
    """Tests that the candidate model with lowest CV RMSE is selected as best."""
    X, y = synthetic_train_data
    candidates = get_model_candidates(random_seed=42)

    comp_df, best_model, best_name, best_metrics = train_and_compare_models(
        X_train=X,
        y_train=y,
        candidates=candidates,
        cv_folds=3,
        random_seed=42,
        enable_mlflow=False,
    )

    assert len(comp_df) == 3
    assert comp_df.iloc[0]["model_name"] == best_name
    # Verify sorted ascending by cv_rmse
    assert comp_df["cv_rmse"].is_monotonic_increasing
    assert best_metrics["model_name"] == best_name


def test_model_serialization_and_reload(synthetic_train_data, tmp_path):
    """Tests saving training outputs and reloading model for inference."""
    X, y = synthetic_train_data
    candidates = get_model_candidates(random_seed=42)
    rf = candidates["random_forest"]
    rf.fit(X, y)

    comp_df = pd.DataFrame([{
        "model_name": "random_forest",
        "cv_rmse": 500.0,
        "cv_mae": 300.0,
        "cv_r2": 0.85,
    }])
    best_metrics = comp_df.iloc[0].to_dict()

    paths = save_training_outputs(
        best_model=rf,
        best_model_name="random_forest",
        comparison_df=comp_df,
        best_metrics=best_metrics,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )

    # Verify files created
    assert Path(paths["best_model_path"]).exists()
    assert Path(paths["comparison_csv_path"]).exists()
    assert Path(paths["metadata_json_path"]).exists()

    # Verify model reload and identical predictions
    reloaded_model = joblib.load(paths["best_model_path"])
    orig_preds = rf.predict(X)
    reloaded_preds = reloaded_model.predict(X)
    np.testing.assert_allclose(orig_preds, reloaded_preds)

    # Verify metadata JSON content
    with open(paths["metadata_json_path"]) as f:
        meta = json.load(f)
    assert meta["selected_model"] == "random_forest"
    assert meta["selection_criterion"] == "lowest_cv_rmse"


@patch("mlflow.sklearn.log_model")
@patch("mlflow.set_tag")
@patch("mlflow.log_metric")
@patch("mlflow.log_params")
@patch("mlflow.log_param")
@patch("mlflow.start_run")
def test_mlflow_logging_mocked(
    mock_start_run,
    mock_log_param,
    mock_log_params,
    mock_log_metric,
    mock_set_tag,
    mock_log_model,
    synthetic_train_data,
):
    """Tests that MLflow logs parameters, metrics, and models for each candidate."""
    mock_run_ctx = MagicMock()
    mock_start_run.return_value.__enter__.return_value = mock_run_ctx

    X, y = synthetic_train_data
    candidates = get_model_candidates(random_seed=42)

    comp_df, best_model, best_name, _ = train_and_compare_models(
        X_train=X,
        y_train=y,
        candidates=candidates,
        cv_folds=3,
        random_seed=42,
        enable_mlflow=True,
        tracking_uri="sqlite:///tmp_mlruns.db",
        experiment_name="Test_Exp",
    )

    # start_run should be called for each candidate (3) + 1 for the best model summary run
    assert mock_start_run.call_count == 4
    assert mock_log_metric.call_count > 0
    assert mock_log_model.call_count == 4


def test_run_training_pipeline_end_to_end():
    """Tests execution of the training pipeline using real processed dataset."""
    results = run_training_pipeline("config/config.yaml")

    assert results["best_model_name"] in ["linear_regression", "random_forest", "gradient_boosting"]
    assert len(results["comparison_table"]) == 3
    assert Path(results["paths"]["best_model_path"]).exists()
    assert Path(results["paths"]["comparison_csv_path"]).exists()
    assert Path(results["paths"]["metadata_json_path"]).exists()
