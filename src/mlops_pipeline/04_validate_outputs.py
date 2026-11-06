"""Stage 04: Validation of Preprocessing Outputs and Artifacts.

Performs integrity checks on generated feature matrices, target vectors,
and serialized preprocessor artifact to verify zero data leakage and numeric consistency.
"""

import sys
import numpy as np
import pandas as pd

from insurance_mlops.features.preprocessor import load_preprocessor
from insurance_mlops.utils.common import get_project_root, load_yaml
from insurance_mlops.utils.logger import get_logger

logger = get_logger("stage_04_validate_outputs")


def validate_preprocessing_outputs(config_path: str = "config/config.yaml") -> bool:
    """Validates all outputs from Stage 03 Preprocessing.

    Checks:
    - Preprocessor artifact existence and loadability.
    - Presence of processed CSV files (X_train, X_test, y_train, y_test).
    - Row count alignment between features and targets.
    - Column dimension matching between train and test.
    - Zero NaN, Null, or Infinite values across all matrices.
    - Standardization check on training features (mean ~ 0, std ~ 1).

    Returns:
        True if all checks pass.

    Raises:
        AssertionError or ValueError on validation failure.
    """
    root = get_project_root()
    config = load_yaml(config_path)

    preproc_cfg = config.get("preprocessing", {})
    preproc_path = root / preproc_cfg.get("preprocessor_path", "models/preprocessor.joblib")
    X_train_path = root / preproc_cfg.get("X_train_path", "data/processed/X_train.csv")
    X_test_path = root / preproc_cfg.get("X_test_path", "data/processed/X_test.csv")
    y_train_path = root / preproc_cfg.get("y_train_path", "data/processed/y_train.csv")
    y_test_path = root / preproc_cfg.get("y_test_path", "data/processed/y_test.csv")

    logger.info("Validating preprocessing artifacts and processed data...")

    # 1. Artifact Verification
    if not preproc_path.exists():
        raise FileNotFoundError(f"Missing preprocessor artifact: {preproc_path}")
    preprocessor = load_preprocessor(preproc_path)
    logger.info(f"✓ Preprocessor artifact verified at: {preproc_path}")

    # 2. File Existence
    for name, p in [
        ("X_train", X_train_path),
        ("X_test", X_test_path),
        ("y_train", y_train_path),
        ("y_test", y_test_path),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required output file: {name} at {p}")
        logger.info(f"✓ Found {name} at: {p}")

    # 3. Load Data
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)

    # 4. Shape & Dimension Checks
    assert len(X_train) == len(y_train), f"X_train rows ({len(X_train)}) != y_train rows ({len(y_train)})"
    assert len(X_test) == len(y_test), f"X_test rows ({len(X_test)}) != y_test rows ({len(y_test)})"
    assert X_train.shape[1] == X_test.shape[1], (
        f"Feature mismatch: X_train cols ({X_train.shape[1]}) != X_test cols ({X_test.shape[1]})"
    )
    logger.info(
        f"✓ Shape parity confirmed: Train=({X_train.shape[0]}x{X_train.shape[1]}), "
        f"Test=({X_test.shape[0]}x{X_test.shape[1]})"
    )

    # 5. Null, NaN, and Infinite Values
    for name, df in [("X_train", X_train), ("X_test", X_test), ("y_train", y_train), ("y_test", y_test)]:
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            raise ValueError(f"{name} contains {null_count} null/NaN values!")
        if np.isinf(df.to_numpy()).any():
            raise ValueError(f"{name} contains infinite values!")
    logger.info("✓ Zero NaN, Null, or Infinite values across all datasets.")

    # 6. Feature Name Parity
    expected_features = list(preprocessor.get_feature_names_out())
    assert list(X_train.columns) == expected_features, "X_train columns do not match preprocessor feature names!"
    assert list(X_test.columns) == expected_features, "X_test columns do not match preprocessor feature names!"
    logger.info(f"✓ Feature names match preprocessor: {expected_features}")

    # 7. Numerical Scaling Sanity Check
    num_cols = [c for c in X_train.columns if c.startswith("num__")]
    for col in num_cols:
        col_mean = X_train[col].mean()
        col_std = X_train[col].std(ddof=0)
        assert abs(col_mean) < 1e-2, f"Column {col} mean ({col_mean:.4f}) is not ~0 (StandardScaler)"
        assert abs(col_std - 1.0) < 1e-2, f"Column {col} std ({col_std:.4f}) is not ~1 (StandardScaler)"
    logger.info("✓ StandardScaler normalization confirmed on training features (mean ~ 0, std ~ 1).")

    # 8. Categorical One-Hot Values Sanity Check
    cat_cols = [c for c in X_train.columns if c.startswith("cat__")]
    for col in cat_cols:
        unique_vals = set(X_train[col].unique())
        assert unique_vals.issubset({0.0, 1.0}), f"One-hot column {col} contains unexpected values: {unique_vals}"
    logger.info("✓ OneHotEncoder binary outputs confirmed (0.0 or 1.0).")

    logger.info("================ All Preprocessing Output Validations PASSED ================")
    return True


def main():
    try:
        validate_preprocessing_outputs()
    except Exception as exc:
        logger.error(f"Stage 04 validation failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
