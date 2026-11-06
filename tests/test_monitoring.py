"""Integration tests for production data drift monitoring and reporting."""

import json
from pathlib import Path
import pandas as pd
import pytest

from insurance_mlops.monitoring.monitor import DataDriftMonitor


@pytest.fixture
def monitor():
    """Initializes production DataDriftMonitor instance."""
    return DataDriftMonitor(config_path="config/config.yaml")


def test_monitor_initialization(monitor):
    """Tests monitor configuration parameters."""
    assert monitor.p_value_threshold == 0.05
    assert monitor.alert_drift_share_threshold == 0.33
    assert len(monitor.detector.numerical_features) == 3
    assert len(monitor.detector.categorical_features) == 3


def test_load_reference_data(monitor):
    """Tests baseline reference data loading without modifications."""
    ref_df = monitor.load_reference_data()
    assert len(ref_df) == 1070
    assert "age" in ref_df.columns
    assert "smoker" in ref_df.columns


def test_run_monitoring_healthy(monitor):
    """Tests monitoring on held-out test split produces healthy status and valid reports."""
    test_df = pd.read_csv("data/processed/test.csv")
    summary = monitor.run_monitoring(current_df=test_df)

    assert summary["status"] == "HEALTHY"
    assert summary["alert_triggered"] is False
    assert summary["total_features_evaluated"] == 6
    assert summary["drift_share"] < monitor.alert_drift_share_threshold

    # Verify JSON report creation and contents
    assert Path(monitor.report_json_path).exists()
    with open(monitor.report_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "HEALTHY"
    assert len(data["feature_drift_results"]) == 6

    # Verify CSV report creation
    assert Path(monitor.report_csv_path).exists()
    df_csv = pd.read_csv(monitor.report_csv_path)
    assert len(df_csv) == 6
    assert "feature_name" in df_csv.columns
    assert "p_value" in df_csv.columns


def test_run_monitoring_drift_alert(monitor):
    """Tests that a heavily perturbed production dataset triggers a drift alert."""
    ref_df = monitor.load_reference_data()

    # Create synthetic shifted batch: age shifted +30, bmi shifted +20, 95% smokers
    shifted_df = ref_df.copy()
    shifted_df["age"] = shifted_df["age"] + 30
    shifted_df["bmi"] = shifted_df["bmi"] + 20.0
    shifted_df["smoker"] = "yes"

    summary = monitor.run_monitoring(current_df=shifted_df, reference_df=ref_df)

    assert summary["status"] == "DRIFT_ALERT"
    assert summary["alert_triggered"] is True
    assert summary["drifted_features_count"] >= 3
    assert "age" in summary["drifted_features"]
    assert "bmi" in summary["drifted_features"]
    assert "smoker" in summary["drifted_features"]


def test_missing_reference_raises_error(tmp_path):
    """Tests that a missing reference dataset path raises FileNotFoundError."""
    dummy_config = tmp_path / "dummy_config.yaml"
    with open(dummy_config, "w") as f:
        f.write("""
monitoring:
  reference_path: "data/non_existent.csv"
  p_value_threshold: 0.05
  alert_drift_share_threshold: 0.33
features:
  numerical: ["age"]
  categorical: ["sex"]
""")
    mon = DataDriftMonitor(config_path=str(dummy_config))
    with pytest.raises(FileNotFoundError):
        mon.load_reference_data()
