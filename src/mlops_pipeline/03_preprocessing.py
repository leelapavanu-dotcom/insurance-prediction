"""Stage 03: Feature Preprocessing Pipeline Orchestration.

Loads raw train and test data, fits ColumnTransformer on training features,
transforms train and test partitions, and saves serialized preprocessor and
processed datasets.
"""

import sys
from insurance_mlops.features.preprocessor import run_preprocessing
from insurance_mlops.utils.logger import get_logger

logger = get_logger("stage_03_preprocessing")


def main():
    try:
        logger.info("Executing Stage 03: Preprocessing and Feature Engineering...")
        results = run_preprocessing(config_path="config/config.yaml")

        logger.info("Stage 03 Results Summary:")
        logger.info(f"  - Preprocessor Artifact: {results['preprocessor_path']}")
        logger.info(f"  - X_train Shape: {results['X_train_shape']}")
        logger.info(f"  - X_test Shape: {results['X_test_shape']}")
        logger.info(f"  - Number of Features: {results['num_features']}")
        logger.info(f"  - Feature List: {results['feature_names']}")
        logger.info("Stage 03 completed successfully.")
    except Exception as exc:
        logger.error(f"Stage 03 failed with error: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
