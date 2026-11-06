"""Stage 07: Quality Gate Enforcement.

Evaluates held-out test metrics against defined quality gate thresholds
in config/quality_gates.yaml (minimum R², maximum RMSE, maximum MAE, overfitting delta).
Outputs decision to reports/quality_gate.json.
"""

import json
import sys

from insurance_mlops.evaluation.evaluator import check_quality_gates
from insurance_mlops.utils.common import get_project_root
from insurance_mlops.utils.logger import get_logger

logger = get_logger("stage_07_quality_gate")


def main():
    try:
        logger.info("Executing Stage 07: Quality Gate Verification...")
        root = get_project_root()
        eval_json_path = root / "reports" / "test_evaluation.json"

        if not eval_json_path.exists():
            raise FileNotFoundError(
                f"Missing test evaluation results at {eval_json_path}. Please execute Stage 06 first."
            )

        with open(eval_json_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        test_metrics = eval_data.get("metrics", {})
        train_r2 = eval_data.get("train_r2")

        gate_results = check_quality_gates(
            test_metrics=test_metrics,
            train_r2=train_r2,
            quality_gates_path="config/quality_gates.yaml",
        )

        logger.info("\n" + "=" * 65)
        logger.info(f"★ QUALITY GATE DECISION: {gate_results['status']}")
        logger.info("=" * 65)
        for check in gate_results.get("checks", []):
            symbol = "✓ PASS" if check["passed"] else "✗ FAIL"
            logger.info(
                f"  [{symbol}] {check['metric']:20}: Actual={check['actual']} "
                f"{check['condition']} Threshold={check['threshold']}"
            )
        logger.info("=" * 65)

        if not gate_results["passed"]:
            logger.error("Quality gate checks FAILED. Model is NOT eligible for production deployment.")
            sys.exit(1)

        logger.info("Quality gate checks PASSED. Model is certified for production deployment.")
        logger.info("Stage 07 completed successfully.")
    except Exception as exc:
        logger.error(f"Stage 07 failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
