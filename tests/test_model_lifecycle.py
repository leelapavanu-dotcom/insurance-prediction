"""Unit tests for Model Lifecycle Management and MLflow Model Registry."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from insurance_mlops.models.registry import register_and_promote_model


def test_promotion_blocked_when_quality_gate_fails(tmp_path):
    """Verifies that a model failing the quality gate is strictly blocked from promotion."""
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    gate_file = reports_dir / "quality_gate.json"

    # Inject a failing quality gate report
    failed_payload = {
        "passed": False,
        "status": "FAIL",
        "failure_reasons": ["R2 below minimum threshold"],
        "checks": [],
    }
    with open(gate_file, "w", encoding="utf-8") as f:
        json.dump(failed_payload, f)

    result = register_and_promote_model("config/config.yaml", alias="champion", check_gate=True)

    assert result["promoted"] is False
    assert result["status"] == "REJECTED_QUALITY_GATE_FAILED"
    assert "reasons" in result


@patch("insurance_mlops.models.registry.mlflow.register_model")
@patch("insurance_mlops.models.registry.MlflowClient")
def test_register_and_promote_model_mocked(mock_client_cls, mock_register_model):
    """Tests successful model registration and alias assignment with mocked MLflow."""
    # Ensure passing quality gate
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    gate_file = reports_dir / "quality_gate.json"

    with open(gate_file, "w", encoding="utf-8") as f:
        json.dump({"passed": True, "status": "PASS", "failure_reasons": []}, f)

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_mv = MagicMock()
    mock_mv.version = "2"
    mock_register_model.return_value = mock_mv

    result = register_and_promote_model("config/config.yaml", alias="champion", check_gate=True)

    assert result["promoted"] is True
    assert result["status"] == "PROMOTED"
    assert result["version"] == "2"
    assert result["alias"] == "champion"
    mock_client.set_registered_model_alias.assert_called_once_with(
        name="insurance_charges_regressor",
        alias="champion",
        version="2",
    )


@patch("insurance_mlops.models.registry.mlflow.register_model")
def test_register_and_promote_fallback_handling(mock_register_model):
    """Tests graceful fallback when MLflow registry service encounters an error."""
    # Ensure passing quality gate
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    gate_file = reports_dir / "quality_gate.json"

    with open(gate_file, "w", encoding="utf-8") as f:
        json.dump({"passed": True, "status": "PASS", "failure_reasons": []}, f)

    # Simulate MLflow database / network failure
    mock_register_model.side_effect = RuntimeError("MLflow Registry DB Connection Timeout")

    result = register_and_promote_model("config/config.yaml", alias="champion", check_gate=True)

    assert result["promoted"] is True
    assert result["status"] == "LOCAL_ARTIFACT_ACTIVE"
    assert "fallback_reason" in result
    assert "MLflow Registry DB Connection Timeout" in result["fallback_reason"]
