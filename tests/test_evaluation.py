"""Unit and integration tests for held-out model evaluation."""

import json
from pathlib import Path
import numpy as np
import pytest

from insurance_mlops.evaluation.evaluator import (
    evaluate_held_out_test,
    evaluate_predictions,
)


def test_evaluate_predictions_numeric():
    """Tests accuracy of metric calculations on controlled arrays."""
    y_true = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    y_pred = np.array([1100.0, 1900.0, 3100.0, 3900.0])

    metrics = evaluate_predictions(y_true, y_pred)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert np.isclose(metrics["mae"], 100.0)
    assert np.isclose(metrics["rmse"], 100.0)
    assert metrics["r2"] > 0.99


def test_evaluate_held_out_test_real_artifacts():
    """Tests held-out evaluation against real serialized model and test split."""
    results = evaluate_held_out_test("config/config.yaml", enable_mlflow=False)

    metrics = results["metrics"]
    assert metrics["r2"] > 0.80, f"Held-out test R2 ({metrics['r2']}) should beat 0.80"
    assert metrics["rmse"] < 5500.0, f"Held-out test RMSE ({metrics['rmse']}) should be < $5,500"
    assert metrics["mae"] < 3200.0

    # Verify report files were created
    assert Path(results["json_path"]).exists()
    assert Path(results["csv_path"]).exists()

    with open(results["json_path"]) as f:
        data = json.load(f)
    assert "metrics" in data
    assert data["test_sample_count"] == 268


def test_missing_model_raises_error(tmp_path):
    """Tests that missing model artifact raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        evaluate_held_out_test("config/non_existent_config.yaml")
