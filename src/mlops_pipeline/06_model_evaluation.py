"""Stage 06: Held-Out Model Evaluation.

Evaluates the selected model strictly on held-out test data (X_test.csv, y_test.csv),
saves metrics to reports/test_evaluation.json and reports/test_evaluation.csv,
and logs the evaluation run to MLflow.
"""

import sys
from insurance_mlops.evaluation.evaluator import evaluate_held_out_test
from insurance_mlops.utils.logger import get_logger

logger = get_logger("stage_06_model_evaluation")


def main():
    try:
        logger.info("Executing Stage 06: Held-Out Model Evaluation...")
        results = evaluate_held_out_test(config_path="config/config.yaml")

        metrics = results["metrics"]
        logger.info("\n" + "=" * 60)
        logger.info("★ HELD-OUT TEST EVALUATION RESULTS:")
        logger.info(f"  - RMSE: ${metrics['rmse']:,.2f}")
        logger.info(f"  - MAE:  ${metrics['mae']:,.2f}")
        logger.info(f"  - R²:   {metrics['r2']:.4f}")
        if results.get("delta_r2") is not None:
            logger.info(f"  - Overfitting Delta R²: {results['delta_r2']:.4f}")
        logger.info("=" * 60)

        logger.info("Reports saved to:")
        logger.info(f"  - JSON: {results['json_path']}")
        logger.info(f"  - CSV:  {results['csv_path']}")
        logger.info("Stage 06 completed successfully.")
    except Exception as exc:
        logger.error(f"Stage 06 evaluation failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
