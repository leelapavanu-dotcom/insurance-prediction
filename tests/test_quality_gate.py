"""Unit tests for Quality Gate threshold evaluation and enforcement."""

from pathlib import Path
import json

from insurance_mlops.evaluation.evaluator import check_quality_gates


def test_quality_gate_pass():
    """Tests that a model exceeding all performance standards passes."""
    test_metrics = {
        "rmse": 4500.0,  # Below 5200.0 threshold
        "mae": 2600.0,   # Below 3000.0 threshold
        "r2": 0.86,      # Above 0.80 threshold
    }
    train_r2 = 0.89     # Overfitting delta = 0.03 <= 0.15

    result = check_quality_gates(test_metrics, train_r2=train_r2, quality_gates_path="config/quality_gates.yaml")

    assert result["passed"] is True
    assert result["status"] == "PASS"
    assert len(result["failure_reasons"]) == 0
    assert len(result["checks"]) == 4


def test_quality_gate_fail_r2():
    """Tests that a model failing the minimum R2 threshold fails the gate."""
    test_metrics = {
        "rmse": 4800.0,
        "mae": 2800.0,
        "r2": 0.72,      # Below 0.80 threshold (violates gate)
    }

    result = check_quality_gates(test_metrics, train_r2=0.85, quality_gates_path="config/quality_gates.yaml")

    assert result["passed"] is False
    assert result["status"] == "FAIL"
    assert any("R² Score check FAILED" in r for r in result["failure_reasons"])


def test_quality_gate_fail_rmse():
    """Tests that a model failing the maximum RMSE threshold fails the gate."""
    test_metrics = {
        "rmse": 6500.0,  # Exceeds 5200.0 threshold (violates gate)
        "mae": 2800.0,
        "r2": 0.82,
    }

    result = check_quality_gates(test_metrics, train_r2=0.85, quality_gates_path="config/quality_gates.yaml")

    assert result["passed"] is False
    assert result["status"] == "FAIL"
    assert any("RMSE check FAILED" in r for r in result["failure_reasons"])


def test_quality_gate_fail_overfitting():
    """Tests that severe overfitting (train R2 - test R2 > 0.15) fails the gate."""
    test_metrics = {
        "rmse": 4600.0,
        "mae": 2600.0,
        "r2": 0.81,
    }
    train_r2 = 0.98     # Delta = 0.17 > 0.15

    result = check_quality_gates(test_metrics, train_r2=train_r2, quality_gates_path="config/quality_gates.yaml")

    assert result["passed"] is False
    assert result["status"] == "FAIL"
    assert any("Overfitting check FAILED" in r for r in result["failure_reasons"])


def test_quality_gate_persistence():
    """Tests that reports/quality_gate.json is created and populated."""
    test_metrics = {"rmse": 4400.0, "mae": 2500.0, "r2": 0.87}
    check_quality_gates(test_metrics, train_r2=0.90, quality_gates_path="config/quality_gates.yaml")

    report_file = Path("reports/quality_gate.json")
    assert report_file.exists()

    with open(report_file) as f:
        data = json.load(f)
    assert "status" in data
    assert "checks" in data
