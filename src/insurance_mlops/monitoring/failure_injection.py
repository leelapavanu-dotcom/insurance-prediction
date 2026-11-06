"""Resilience and Failure Injection Testing Suite.

Systematically tests 10 critical failure modes across the MLOps pipeline
using isolated temporary environments and mocks to verify robust error handling,
safe fallbacks, and zero silent failures without altering production artifacts.
"""

from typing import Any, Dict, List
from fastapi import HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel

from insurance_mlops.data.loader import load_raw_data
from insurance_mlops.data.validator import validate_schema
from insurance_mlops.evaluation.evaluator import check_quality_gates
from insurance_mlops.inference.predictor import InsurancePredictor
from insurance_mlops.models.registry import register_and_promote_model
from insurance_mlops.monitoring.monitor import DataDriftMonitor
from insurance_mlops.utils.common import load_yaml
from insurance_mlops.utils.logger import get_logger

logger = get_logger(__name__)


class ScenarioResult(BaseModel):
    """Result of an individual failure injection test."""

    scenario_id: int
    name: str
    expected_error: str
    actual_behavior: str
    resilient: bool
    details: str


class FailureInjectionSimulator:
    """Orchestrates safe failure injection simulations across the MLOps pipeline."""

    def __init__(self):
        self.results: List[ScenarioResult] = []

    def simulate_1_missing_raw_dataset(self) -> ScenarioResult:
        """Scenario 1: Missing raw dataset file."""
        scenario_name = "Missing Raw Dataset"
        expected = "FileNotFoundError"
        try:
            load_raw_data("data/raw/completely_missing_dataset_file.csv")
            resilient = False
            actual = "Unexpectedly succeeded without raising error."
            details = "Silent failure: Function did not fail when file was absent."
        except FileNotFoundError as exc:
            resilient = True
            actual = f"Caught expected FileNotFoundError: {exc}"
            details = "Handled gracefully: Raised FileNotFoundError with clear message."
        except Exception as exc:
            resilient = False
            actual = f"Caught unexpected exception type: {type(exc).__name__}: {exc}"
            details = "Unexpected error type."

        return ScenarioResult(
            scenario_id=1,
            name=scenario_name,
            expected_error=expected,
            actual_behavior=actual,
            resilient=resilient,
            details=details,
        )

    def simulate_2_invalid_data_schema(self) -> ScenarioResult:
        """Scenario 2: Data schema violation (corrupted columns and invalid values)."""
        scenario_name = "Invalid Data Schema"
        expected = "ValueError"
        bad_df = pd.DataFrame({
            "age": [12, 150, -5],  # Out of valid [18, 100] range
            "sex": ["alien", "robot", "unknown"],  # Invalid categories
            "bmi": [5.0, 95.0, -1.0],  # Out of [10, 70] range
            "children": [-2, 50, 100],  # Out of range
            "smoker": ["sometimes", "maybe", "never"],  # Invalid categories
            "region": ["atlantis", "mars", "moon"],  # Invalid regions
            "charges": [1000.0, 2000.0, 3000.0],
        })
        try:
            validate_schema(bad_df, is_training=True)
            resilient = False
            actual = "Validation succeeded unexpectedly."
            details = "Silent failure: Corrupt data bypassed schema checks."
        except ValueError as exc:
            resilient = True
            actual = f"Caught expected ValueError: {exc}"
            details = "Handled gracefully: Schema enforcement blocked invalid data."
        except Exception as exc:
            resilient = False
            actual = f"Caught unexpected exception: {type(exc).__name__}: {exc}"
            details = "Unexpected error type."

        return ScenarioResult(
            scenario_id=2,
            name=scenario_name,
            expected_error=expected,
            actual_behavior=actual,
            resilient=resilient,
            details=details,
        )

    def simulate_3_missing_model_artifact(self) -> ScenarioResult:
        """Scenario 3: Missing trained model artifact."""
        scenario_name = "Missing Model Artifact"
        expected = "FileNotFoundError"
        try:
            InsurancePredictor(model_path="models/non_existent_model_file.joblib")
            resilient = False
            actual = "Predictor initialized unexpectedly without model."
            details = "Silent failure: Missing model did not trigger error."
        except FileNotFoundError as exc:
            resilient = True
            actual = f"Caught expected FileNotFoundError: {exc}"
            details = "Handled gracefully: Initialization blocked due to missing model artifact."
        except Exception as exc:
            resilient = False
            actual = f"Caught unexpected exception: {type(exc).__name__}: {exc}"
            details = "Unexpected error type."

        return ScenarioResult(
            scenario_id=3,
            name=scenario_name,
            expected_error=expected,
            actual_behavior=actual,
            resilient=resilient,
            details=details,
        )

    def simulate_4_missing_preprocessing_artifact(self) -> ScenarioResult:
        """Scenario 4: Missing preprocessor artifact."""
        scenario_name = "Missing Preprocessor Artifact"
        expected = "FileNotFoundError"
        try:
            InsurancePredictor(preprocessor_path="models/non_existent_preprocessor.joblib")
            resilient = False
            actual = "Predictor initialized unexpectedly without preprocessor."
            details = "Silent failure: Missing preprocessor did not trigger error."
        except FileNotFoundError as exc:
            resilient = True
            actual = f"Caught expected FileNotFoundError: {exc}"
            details = "Handled gracefully: Initialization blocked due to missing preprocessor artifact."
        except Exception as exc:
            resilient = False
            actual = f"Caught unexpected exception: {type(exc).__name__}: {exc}"
            details = "Unexpected error type."

        return ScenarioResult(
            scenario_id=4,
            name=scenario_name,
            expected_error=expected,
            actual_behavior=actual,
            resilient=resilient,
            details=details,
        )

    def simulate_5_invalid_inference_input(self) -> ScenarioResult:
        """Scenario 5: Malformed inference payload (out-of-range values and bad keys)."""
        scenario_name = "Invalid Inference Input"
        expected = "ValueError"
        try:
            predictor = InsurancePredictor()
            bad_records = [{"age": 250, "sex": "invalid_gender", "bmi": -10.0}]
            predictor.predict_records(bad_records)
            resilient = False
            actual = "Prediction succeeded on corrupt input."
            details = "Silent failure: Malformed inference request was not rejected."
        except ValueError as exc:
            resilient = True
            actual = f"Caught expected ValueError: {exc}"
            details = "Handled gracefully: Runtime validation rejected invalid payload."
        except Exception as exc:
            resilient = False
            actual = f"Caught unexpected exception: {type(exc).__name__}: {exc}"
            details = "Unexpected error type."

        return ScenarioResult(
            scenario_id=5,
            name=scenario_name,
            expected_error=expected,
            actual_behavior=actual,
            resilient=resilient,
            details=details,
        )

    def simulate_6_quality_gate_failure(self) -> ScenarioResult:
        """Scenario 6: Quality gate triggers failure on substandard model metrics."""
        scenario_name = "Quality Gate Rejection"
        expected = "Gate Rejection (status=FAIL, passed=False)"
        from pathlib import Path

        q_path = Path("reports/quality_gate.json")
        orig_q = q_path.read_text(encoding="utf-8") if q_path.exists() else None

        try:
            substandard_metrics = {
                "rmse": 18500.0,  # Far above 5200.0 threshold
                "mae": 12000.0,   # Far above 3000.0 threshold
                "r2": 0.35,       # Far below 0.80 threshold
            }
            res = check_quality_gates(
                test_metrics=substandard_metrics,
                train_r2=0.95,  # High overfitting delta (0.60 > 0.15)
                quality_gates_path="config/quality_gates.yaml",
            )
            if not res["passed"] and res["status"] == "FAIL" and len(res["failure_reasons"]) > 0:
                resilient = True
                actual = (
                    f"Gate correctly failed with {len(res['failure_reasons'])} violations: "
                    f"{res['failure_reasons']}"
                )
                details = "Handled gracefully: Substandard model blocked from deployment."
            else:
                resilient = False
                actual = f"Gate unexpectedly returned status={res.get('status')}"
                details = "Silent failure: Substandard model erroneously passed quality gates."
        finally:
            if orig_q is not None:
                q_path.write_text(orig_q, encoding="utf-8")

        return ScenarioResult(
            scenario_id=6,
            name=scenario_name,
            expected_error=expected,
            actual_behavior=actual,
            resilient=resilient,
            details=details,
        )

    def simulate_7_data_drift_alert(self) -> ScenarioResult:
        """Scenario 7: Extreme distribution shift triggers alert."""
        scenario_name = "Data Drift Alert Trigger"
        expected = "Drift Alert Triggered (alert_triggered=True)"
        from pathlib import Path

        json_path = Path("reports/drift_report.json")
        csv_path = Path("reports/drift_report.csv")
        orig_json = json_path.read_text(encoding="utf-8") if json_path.exists() else None
        orig_csv = csv_path.read_text(encoding="utf-8") if csv_path.exists() else None

        try:
            # Create heavily drifted test data
            drifted_df = pd.DataFrame({
                "age": np.random.normal(75, 2, 200),  # Age shifted from ~39 to 75
                "bmi": np.random.normal(55, 3, 200),  # BMI shifted from ~30 to 55
                "children": np.random.choice([4, 5], 200),
                "sex": ["female"] * 200,
                "smoker": ["yes"] * 200,  # 100% smokers vs ~20% baseline
                "region": ["northeast"] * 200,
            })
            monitor = DataDriftMonitor()
            res = monitor.run_monitoring(current_df=drifted_df)
            if res["alert_triggered"] and res["status"] == "DRIFT_ALERT" and res["drift_share"] >= 0.33:
                resilient = True
                actual = (
                    f"Alert triggered: {res['drifted_features_count']}/6 features drifted "
                    f"({res['drift_share']:.1%})"
                )
                details = "Handled gracefully: Statistical shift flagged and alert triggered."
            else:
                resilient = False
                actual = (
                    f"Alert not triggered (status={res.get('status')}, "
                    f"drifted={res.get('drifted_features_count')})"
                )
                details = "Silent failure: Extreme drift was not flagged."
        finally:
            if orig_json is not None:
                json_path.write_text(orig_json, encoding="utf-8")
            if orig_csv is not None:
                csv_path.write_text(orig_csv, encoding="utf-8")

        return ScenarioResult(
            scenario_id=7,
            name=scenario_name,
            expected_error=expected,
            actual_behavior=actual,
            resilient=resilient,
            details=details,
        )

    def simulate_8_api_artifact_loading_failure(self) -> ScenarioResult:
        """Scenario 8: API service handles unavailable model gracefully (HTTP 503)."""
        scenario_name = "API Artifact Unavailable"
        expected = "HTTPException (Status 503)"
        from insurance_mlops.inference import api

        # Temporarily point predictor to non-existent location
        original_predictor = api._predictor
        api._predictor = None
        try:
            # Monkeypatch InsurancePredictor to simulate artifact failure
            def mock_init():
                raise FileNotFoundError("Simulated missing best_model.joblib")

            saved_cls = api.InsurancePredictor
            api.InsurancePredictor = mock_init
            try:
                api.get_predictor()
                resilient = False
                actual = "API initialized predictor unexpectedly."
                details = "Silent failure: Missing artifact did not raise HTTP 503."
            except HTTPException as http_exc:
                if http_exc.status_code == 503:
                    resilient = True
                    actual = f"Caught expected HTTPException 503: {http_exc.detail}"
                    details = "Handled gracefully: Returned HTTP 503 Service Unavailable."
                else:
                    resilient = False
                    actual = f"Caught HTTPException with status {http_exc.status_code}"
                    details = "Unexpected status code."
            finally:
                api.InsurancePredictor = saved_cls
        finally:
            api._predictor = original_predictor

        return ScenarioResult(
            scenario_id=8,
            name=scenario_name,
            expected_error=expected,
            actual_behavior=actual,
            resilient=resilient,
            details=details,
        )

    def simulate_9_mlflow_registry_unavailable(self) -> ScenarioResult:
        """Scenario 9: Model registry handles gate check failure safely."""
        scenario_name = "Model Registry Gate Enforcement"
        expected = "Promotion Blocked (status=REJECTED_QUALITY_GATE_FAILED)"
        from pathlib import Path
        import json

        gate_path = Path("reports/quality_gate.json")
        lifecycle_path = Path("reports/model_lifecycle.json")

        original_gate_content = gate_path.read_text(encoding="utf-8") if gate_path.exists() else None
        original_lifecycle_content = lifecycle_path.read_text(encoding="utf-8") if lifecycle_path.exists() else None

        try:
            gate_path.parent.mkdir(parents=True, exist_ok=True)
            failed_gate = {
                "passed": False,
                "status": "FAIL",
                "failure_reasons": ["R2 below 0.80 minimum threshold", "RMSE exceeds 5200.0"],
                "checks": [],
            }
            with open(gate_path, "w", encoding="utf-8") as f:
                json.dump(failed_gate, f)

            res = register_and_promote_model(config_path="config/config.yaml", check_gate=True)
            if not res.get("promoted", True) and "REJECTED" in res.get("status", ""):
                resilient = True
                actual = f"Promotion blocked safely: status={res.get('status')}"
                details = "Handled gracefully: Failed quality gate prevented model promotion."
            else:
                resilient = False
                actual = f"Model was promoted despite failed gate: {res}"
                details = "Silent failure: Model promoted without passing quality gate."
        finally:
            # Restore original artifacts cleanly
            if original_gate_content is not None:
                gate_path.write_text(original_gate_content, encoding="utf-8")
            if original_lifecycle_content is not None:
                lifecycle_path.write_text(original_lifecycle_content, encoding="utf-8")

        return ScenarioResult(
            scenario_id=9,
            name=scenario_name,
            expected_error=expected,
            actual_behavior=actual,
            resilient=resilient,
            details=details,
        )

        return ScenarioResult(
            scenario_id=9,
            name=scenario_name,
            expected_error=expected,
            actual_behavior=actual,
            resilient=resilient,
            details=details,
        )

    def simulate_10_invalid_configuration(self) -> ScenarioResult:
        """Scenario 10: Missing or corrupted YAML configuration file."""
        scenario_name = "Invalid Configuration File"
        expected = "FileNotFoundError"
        try:
            load_yaml("config/completely_non_existent_config.yaml")
            resilient = False
            actual = "load_yaml succeeded without error on missing file."
            details = "Silent failure: Missing config file did not raise error."
        except FileNotFoundError as exc:
            resilient = True
            actual = f"Caught expected FileNotFoundError: {exc}"
            details = "Handled gracefully: Configuration loader rejected missing file."
        except Exception as exc:
            resilient = False
            actual = f"Caught unexpected exception: {type(exc).__name__}: {exc}"
            details = "Unexpected error type."

        return ScenarioResult(
            scenario_id=10,
            name=scenario_name,
            expected_error=expected,
            actual_behavior=actual,
            resilient=resilient,
            details=details,
        )

    def run_all_simulations(self) -> Dict[str, Any]:
        """Runs all 10 failure injection scenarios sequentially."""
        logger.info("Starting Execution of 10 Failure Injection Scenarios...")
        simulations = [
            self.simulate_1_missing_raw_dataset,
            self.simulate_2_invalid_data_schema,
            self.simulate_3_missing_model_artifact,
            self.simulate_4_missing_preprocessing_artifact,
            self.simulate_5_invalid_inference_input,
            self.simulate_6_quality_gate_failure,
            self.simulate_7_data_drift_alert,
            self.simulate_8_api_artifact_loading_failure,
            self.simulate_9_mlflow_registry_unavailable,
            self.simulate_10_invalid_configuration,
        ]

        self.results = []
        for sim in simulations:
            result = sim()
            self.results.append(result)
            symbol = "✓ PASS" if result.resilient else "✗ FAIL"
            logger.info(f"[{symbol}] Scenario {result.scenario_id}: {result.name} -> {result.details}")

        total = len(self.results)
        passed = sum(1 for r in self.results if r.resilient)
        all_resilient = (passed == total)

        summary = {
            "total_scenarios": total,
            "resilient_count": passed,
            "failed_count": total - passed,
            "all_resilient": all_resilient,
            "status": "RESILIENT" if all_resilient else "VULNERABILITIES_FOUND",
            "results": [r.model_dump() for r in self.results],
        }
        return summary
