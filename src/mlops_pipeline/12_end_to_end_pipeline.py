"""Stage 12: Master End-to-End MLOps Pipeline Orchestrator.

Orchestrates the entire lifecycle sequentially:
1. Data Ingestion & Pydantic Schema Validation
2. Feature Preprocessing (ColumnTransformer & Scaler)
3. Preprocessing Output Integrity Checks
4. Model Training, 5-Fold CV & MLflow Tracking
5. Held-Out Test Evaluation
6. Quality Gate Threshold Verification
7. Model Registry & Champion Alias Promotion
8. Batch Production Inference
9. Statistical Data Drift Monitoring
10. Failure Injection & Resilience Verification
11. Final System State Validation

Tracks stage execution times, stops on critical failure, and produces
reports/end_to_end_execution_summary.json.
"""

from datetime import datetime, timezone
import importlib
import json
import sys
import time
from typing import Any, Dict, List
import pandas as pd

from insurance_mlops.data.loader import run_data_pipeline
from insurance_mlops.evaluation.evaluator import check_quality_gates, evaluate_held_out_test
from insurance_mlops.features.preprocessor import run_preprocessing
from insurance_mlops.inference.predictor import predict_batch_csv
from insurance_mlops.models.registry import register_and_promote_model
from insurance_mlops.models.trainer import run_training_pipeline
from insurance_mlops.monitoring.failure_injection import FailureInjectionSimulator
from insurance_mlops.monitoring.monitor import DataDriftMonitor
from insurance_mlops.utils.common import get_project_root, load_yaml
from insurance_mlops.utils.logger import get_logger

logger = get_logger("stage_12_master_pipeline")


class StageResult:
    """Encapsulates the execution result of an individual pipeline stage."""

    def __init__(self, stage_num: int, name: str, status: str, duration_sec: float, details: str):
        self.stage_num = stage_num
        self.name = name
        self.status = status
        self.duration_sec = duration_sec
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_num": self.stage_num,
            "name": self.name,
            "status": self.status,
            "duration_sec": round(self.duration_sec, 2),
            "details": self.details,
        }


def run_master_pipeline(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Executes all 11 stages of the MLOps lifecycle sequentially.

    Args:
        config_path: Relative or absolute path to configuration YAML.

    Returns:
        Structured dictionary containing stage timings, final status, and metrics.
    """
    logger.info("=" * 80)
    logger.info("★ INITIATING MASTER END-TO-END MLOPS PIPELINE ★")
    logger.info("=" * 80)

    start_time_all = time.time()
    stages: List[StageResult] = []
    root = get_project_root()
    config = load_yaml(config_path)

    def execute_stage(stage_num: int, name: str, fn, *args, **kwargs) -> Any:
        logger.info(f"\n>>> Running Stage {stage_num:02d}: {name}...")
        t0 = time.time()
        try:
            result = fn(*args, **kwargs)
            dt = time.time() - t0
            summary_msg = f"Completed in {dt:.2f}s"
            logger.info(f"✓ Stage {stage_num:02d} [{name}] SUCCESS: {summary_msg}")
            stages.append(StageResult(stage_num, name, "SUCCESS", dt, summary_msg))
            return result
        except Exception as exc:
            dt = time.time() - t0
            err_msg = f"Failed after {dt:.2f}s: {exc}"
            logger.error(f"✗ Stage {stage_num:02d} [{name}] FAILED: {err_msg}", exc_info=True)
            stages.append(StageResult(stage_num, name, "FAILED", dt, err_msg))
            raise exc

    try:
        # Stage 01/02: Data Ingestion & Pydantic Schema Validation
        execute_stage(
            1, "Data Ingestion & Validation",
            run_data_pipeline, config_path=config_path
        )

        # Stage 03: Feature Preprocessing
        execute_stage(
            2, "Feature Preprocessing",
            run_preprocessing, config_path=config_path
        )

        # Stage 04: Validate Preprocessing Outputs
        mod_04 = importlib.import_module("mlops_pipeline.04_validate_outputs")
        execute_stage(
            3, "Preprocessing Outputs Validation",
            mod_04.validate_preprocessing_outputs, config_path=config_path
        )

        # Stage 05: Model Training, CV Comparison & MLflow Logging
        execute_stage(
            4, "Model Training & CV Comparison",
            run_training_pipeline, config_path=config_path
        )

        # Stage 06: Held-Out Model Evaluation
        eval_res = execute_stage(
            5, "Held-Out Test Evaluation",
            evaluate_held_out_test, config_path=config_path, enable_mlflow=True
        )

        # Stage 07: Quality Gate Enforcement
        quality_res = execute_stage(
            6, "Quality Gate Enforcement",
            check_quality_gates,
            test_metrics=eval_res["metrics"],
            train_r2=eval_res.get("train_r2"),
            quality_gates_path="config/quality_gates.yaml",
        )
        if not quality_res["passed"]:
            raise ValueError(f"Quality gate rejected model: {quality_res['failure_reasons']}")

        # Stage 08: Model Registry & Alias Promotion
        execute_stage(
            7, "MLflow Model Registry & Promotion",
            register_and_promote_model, config_path=config_path, alias="champion", check_gate=True
        )

        # Stage 09: Batch Production Inference
        sample_input = root / "data" / "processed" / "sample_inference_input.csv"
        batch_output = root / "data" / "processed" / "batch_predictions.csv"
        execute_stage(
            8, "Batch Production Inference",
            predict_batch_csv, input_csv_path=sample_input, output_csv_path=batch_output
        )

        # Stage 10: Statistical Data Drift Monitoring
        def run_drift():
            test_data_path = root / config.get("preprocessing", {}).get("test_path", "data/processed/test.csv")
            test_df = pd.read_csv(test_data_path)
            monitor = DataDriftMonitor(config_path=config_path)
            return monitor.run_monitoring(current_df=test_df)

        execute_stage(9, "Statistical Data Drift Monitoring", run_drift)

        # Stage 11: Failure Injection & Resilience Verification
        def run_resilience():
            sim = FailureInjectionSimulator()
            res = sim.run_all_simulations()
            if not res["all_resilient"]:
                raise RuntimeError("One or more resilience scenarios failed.")
            return res

        execute_stage(10, "Failure Injection & Resilience Verification", run_resilience)

        total_duration = time.time() - start_time_all
        master_status = "ALL_STAGES_PASSED"

    except Exception as fatal_exc:
        total_duration = time.time() - start_time_all
        master_status = f"FAILED: {fatal_exc}"
        logger.error(f"Master pipeline halted due to critical error: {fatal_exc}")

    # Build final report
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": master_status,
        "total_duration_sec": round(total_duration, 2),
        "total_stages": len(stages),
        "successful_stages": sum(1 for s in stages if s.status == "SUCCESS"),
        "failed_stages": sum(1 for s in stages if s.status == "FAILED"),
        "stages": [s.to_dict() for s in stages],
    }

    # Save summary to reports/
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "end_to_end_execution_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print final presentation table
    logger.info("\n" + "=" * 80)
    logger.info("★ MASTER PIPELINE EXECUTION SUMMARY ★")
    logger.info(f"  - Overall Status:   {summary['overall_status']}")
    logger.info(f"  - Total Execution:  {summary['total_duration_sec']}s")
    logger.info(f"  - Stages Passed:     {summary['successful_stages']}/{summary['total_stages']}")
    logger.info(f"  - Summary Report:    {summary_path}")
    logger.info("=" * 80)

    df_stages = pd.DataFrame([s.to_dict() for s in stages])
    print("\nStage Breakdown Table:")
    print(df_stages[["stage_num", "name", "status", "duration_sec", "details"]].to_string(index=False))

    return summary


def main():
    summary = run_master_pipeline()
    if summary["failed_stages"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
