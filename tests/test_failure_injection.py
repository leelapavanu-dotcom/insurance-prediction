"""Unit tests for pipeline resilience and failure injection scenarios."""

from insurance_mlops.monitoring.failure_injection import FailureInjectionSimulator


def test_scenario_1_missing_raw_dataset():
    """Verifies that missing raw dataset raises FileNotFoundError."""
    sim = FailureInjectionSimulator()
    res = sim.simulate_1_missing_raw_dataset()
    assert res.resilient is True
    assert res.scenario_id == 1
    assert "FileNotFoundError" in res.actual_behavior


def test_scenario_2_invalid_data_schema():
    """Verifies that data schema violations raise ValueError."""
    sim = FailureInjectionSimulator()
    res = sim.simulate_2_invalid_data_schema()
    assert res.resilient is True
    assert res.scenario_id == 2
    assert "ValueError" in res.actual_behavior


def test_scenario_3_missing_model_artifact():
    """Verifies that missing model artifact raises FileNotFoundError during initialization."""
    sim = FailureInjectionSimulator()
    res = sim.simulate_3_missing_model_artifact()
    assert res.resilient is True
    assert res.scenario_id == 3
    assert "FileNotFoundError" in res.actual_behavior


def test_scenario_4_missing_preprocessing_artifact():
    """Verifies that missing preprocessor artifact raises FileNotFoundError."""
    sim = FailureInjectionSimulator()
    res = sim.simulate_4_missing_preprocessing_artifact()
    assert res.resilient is True
    assert res.scenario_id == 4
    assert "FileNotFoundError" in res.actual_behavior


def test_scenario_5_invalid_inference_input():
    """Verifies that out-of-boundary inference inputs raise ValueError."""
    sim = FailureInjectionSimulator()
    res = sim.simulate_5_invalid_inference_input()
    assert res.resilient is True
    assert res.scenario_id == 5
    assert "ValueError" in res.actual_behavior


def test_scenario_6_quality_gate_failure():
    """Verifies that substandard metrics result in FAIL status."""
    sim = FailureInjectionSimulator()
    res = sim.simulate_6_quality_gate_failure()
    assert res.resilient is True
    assert res.scenario_id == 6
    assert "Gate correctly failed" in res.actual_behavior


def test_scenario_7_data_drift_alert():
    """Verifies that severe distribution shifts trigger an alert."""
    sim = FailureInjectionSimulator()
    res = sim.simulate_7_data_drift_alert()
    assert res.resilient is True
    assert res.scenario_id == 7
    assert "Alert triggered" in res.actual_behavior


def test_scenario_8_api_artifact_loading_failure():
    """Verifies that missing artifacts trigger HTTP 503 Service Unavailable."""
    sim = FailureInjectionSimulator()
    res = sim.simulate_8_api_artifact_loading_failure()
    assert res.resilient is True
    assert res.scenario_id == 8
    assert "HTTPException 503" in res.actual_behavior


def test_scenario_9_mlflow_registry_unavailable():
    """Verifies that models failing quality gates cannot be promoted."""
    sim = FailureInjectionSimulator()
    res = sim.simulate_9_mlflow_registry_unavailable()
    assert res.resilient is True
    assert res.scenario_id == 9
    assert "Promotion blocked safely" in res.actual_behavior


def test_scenario_10_invalid_configuration():
    """Verifies that missing config YAML raises FileNotFoundError."""
    sim = FailureInjectionSimulator()
    res = sim.simulate_10_invalid_configuration()
    assert res.resilient is True
    assert res.scenario_id == 10
    assert "FileNotFoundError" in res.actual_behavior


def test_full_failure_injection_suite_resilience():
    """Verifies that all 10 simulations pass with 100% resilience."""
    sim = FailureInjectionSimulator()
    summary = sim.run_all_simulations()
    assert summary["all_resilient"] is True
    assert summary["total_scenarios"] == 10
    assert summary["resilient_count"] == 10
    assert summary["failed_count"] == 0
    assert summary["status"] == "RESILIENT"
