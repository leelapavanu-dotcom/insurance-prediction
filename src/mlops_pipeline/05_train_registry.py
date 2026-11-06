"""Stage 05: Model Training, Comparison, and MLflow Tracking.

Loads preprocessed training data, evaluates Linear Regression, Random Forest,
and Gradient Boosting using 5-Fold Cross Validation, logs runs to MLflow,
selects the best model based on lowest RMSE, and serializes the winning artifact.
"""

import sys
import pandas as pd
from insurance_mlops.models.trainer import run_training_pipeline
from insurance_mlops.utils.logger import get_logger

logger = get_logger("stage_05_train_registry")


def main():
    try:
        logger.info("Executing Stage 05: Model Training, Comparison, and MLflow Tracking...")
        results = run_training_pipeline(config_path="config/config.yaml")

        best_model = results["best_model_name"]
        metrics = results["best_metrics"]
        paths = results["paths"]

        logger.info("\n" + "=" * 65)
        logger.info(f"★ SELECTED BEST MODEL: {best_model.upper()}")
        logger.info(f"  - CV RMSE: ${metrics['cv_rmse']:,.2f} (± ${metrics['cv_rmse_std']:,.2f})")
        logger.info(f"  - CV MAE:  ${metrics['cv_mae']:,.2f}")
        logger.info(f"  - CV R²:   {metrics['cv_r2']:.4f}")
        logger.info(f"  - Full Train RMSE: ${metrics['train_rmse']:,.2f}")
        logger.info(f"  - Full Train R²:   {metrics['train_r2']:.4f}")
        logger.info("=" * 65)

        logger.info("\nModel Comparison Table:")
        comp_df = pd.DataFrame(results["comparison_table"])
        print(comp_df.to_string(index=False))

        logger.info("\nArtifacts saved successfully:")
        logger.info(f"  - Model:      {paths['best_model_path']}")
        logger.info(f"  - Comparison: {paths['comparison_csv_path']}")
        logger.info(f"  - Metadata:   {paths['metadata_json_path']}")
        logger.info("Stage 05 completed successfully.")
    except Exception as exc:
        logger.error(f"Stage 05 failed with error: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
