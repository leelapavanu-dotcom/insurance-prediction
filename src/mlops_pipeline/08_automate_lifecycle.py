"""Stage 08: Model Lifecycle Management & Model Registry Automation.

Verifies quality gate passage, registers the model in MLflow Model Registry,
and assigns the 'champion' alias for production serving.
Outputs lifecycle report to reports/model_lifecycle.json.
"""

import sys
from insurance_mlops.models.registry import register_and_promote_model
from insurance_mlops.utils.logger import get_logger

logger = get_logger("stage_08_automate_lifecycle")


def main():
    try:
        logger.info("Executing Stage 08: Model Lifecycle & Registry Management...")
        lifecycle_report = register_and_promote_model(
            config_path="config/config.yaml",
            alias="champion",
            check_gate=True,
        )

        logger.info("\n" + "=" * 65)
        logger.info(f"★ LIFECYCLE STATUS: {lifecycle_report['status']}")
        logger.info(f"  - Registered Name: {lifecycle_report.get('registered_model_name')}")
        logger.info(f"  - Model Version:   {lifecycle_report.get('version')}")
        logger.info(f"  - Assigned Alias:  @{lifecycle_report.get('alias')}")
        logger.info(f"  - Architecture:    {lifecycle_report.get('architecture')}")
        logger.info(f"  - Quality Gate:    {lifecycle_report.get('quality_gate')}")
        logger.info("=" * 65)

        if not lifecycle_report.get("promoted", False):
            logger.error("Model promotion could not be completed.")
            sys.exit(1)

        logger.info("Stage 08 completed successfully. Model is promoted as 'champion'.")
    except Exception as exc:
        logger.error(f"Stage 08 failed with error: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
