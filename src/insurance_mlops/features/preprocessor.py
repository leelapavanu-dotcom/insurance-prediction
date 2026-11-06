"""Feature preprocessing and transformation pipeline."""

from pathlib import Path
from typing import Any, Dict, List, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from insurance_mlops.utils.common import get_project_root, load_yaml
from insurance_mlops.utils.logger import get_logger

logger = get_logger(__name__)


def build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
    scaler_type: str = "standard",
) -> ColumnTransformer:
    """Builds an unfitted ColumnTransformer for numerical and categorical features.

    Args:
        numerical_features: List of continuous/discrete numeric column names.
        categorical_features: List of categorical column names.
        scaler_type: 'standard' for StandardScaler, 'robust' for RobustScaler.

    Returns:
        Unfitted ColumnTransformer instance.
    """
    logger.info(
        f"Building preprocessor: numerical={numerical_features}, "
        f"categorical={categorical_features}, scaler={scaler_type}"
    )

    if scaler_type.lower() == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numerical_features),
            ("cat", encoder, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Fits the ColumnTransformer strictly on the training dataset.

    Args:
        preprocessor: Unfitted ColumnTransformer.
        X_train: Training features DataFrame.

    Returns:
        Fitted ColumnTransformer.
    """
    logger.info(f"Fitting preprocessor on training data ({X_train.shape[0]} rows, {X_train.shape[1]} cols)...")
    preprocessor.fit(X_train)
    logger.info("Preprocessor fitting complete. No test data was exposed.")
    return preprocessor


def transform_data(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Transforms a feature DataFrame using a fitted ColumnTransformer.

    Validates that transformed data contains no NaN or infinite values
    and preserves human-readable feature column names.

    Args:
        preprocessor: Fitted ColumnTransformer.
        X: Feature DataFrame to transform.

    Returns:
        Transformed pandas DataFrame with preserved feature names.

    Raises:
        ValueError: If transformed values contain NaN or Inf.
    """
    transformed_array = preprocessor.transform(X)

    # Validate numeric integrity
    if np.isnan(transformed_array).any():
        err = "Transformed dataset contains NaN values!"
        logger.error(err)
        raise ValueError(err)

    if np.isinf(transformed_array).any():
        err = "Transformed dataset contains infinite values!"
        logger.error(err)
        raise ValueError(err)

    feature_names = preprocessor.get_feature_names_out()
    transformed_df = pd.DataFrame(
        transformed_array,
        columns=feature_names,
        index=X.index,
    )
    logger.debug(f"Transformed {len(X)} rows into {len(feature_names)} features.")
    return transformed_df


def save_preprocessor(
    preprocessor: ColumnTransformer,
    output_path: Union[str, Path],
) -> Path:
    """Serializes the fitted preprocessor to disk using joblib.

    Args:
        preprocessor: Fitted ColumnTransformer.
        output_path: Destination path.

    Returns:
        Path to saved artifact.
    """
    path = Path(output_path)
    if not path.is_absolute():
        path = get_project_root() / path

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    logger.info(f"Saved fitted preprocessor artifact to: {path}")
    return path


def load_preprocessor(
    path: Union[str, Path],
) -> ColumnTransformer:
    """Loads a serialized preprocessor from disk.

    Args:
        path: Path to serialized artifact.

    Returns:
        Fitted ColumnTransformer.
    """
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = get_project_root() / file_path

    if not file_path.exists():
        err = f"Preprocessor artifact not found at: {file_path}"
        logger.error(err)
        raise FileNotFoundError(err)

    preprocessor = joblib.load(file_path)
    logger.info(f"Loaded preprocessor successfully from: {file_path}")
    return preprocessor


def run_preprocessing(
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """Executes the full preprocessing pipeline:

    1. Loads train.csv and test.csv from data/processed/.
    2. Separates target column ('charges') from features.
    3. Builds ColumnTransformer (StandardScaler + OneHotEncoder).
    4. Fits preprocessor ONLY on training data to prevent leakage.
    5. Transforms both train and test partitions.
    6. Validates dimension matching and absence of NaN/Inf values.
    7. Saves fitted preprocessor and transformed datasets.

    Args:
        config_path: Path to configuration YAML.

    Returns:
        Dictionary containing summary statistics, feature names, and paths.
    """
    logger.info("================ Starting Preprocessing Pipeline ================")
    root = get_project_root()
    config = load_yaml(config_path)

    # Paths and configurations
    train_path = root / config["data"]["train_path"]
    test_path = root / config["data"]["test_path"]
    target_col = config["features"]["target"]
    num_cols = config["features"]["numerical"]
    cat_cols = config["features"]["categorical"]

    preproc_cfg = config.get("preprocessing", {})
    scaler_type = preproc_cfg.get("scaler", "standard")
    preproc_save_path = root / preproc_cfg.get("preprocessor_path", "models/preprocessor.joblib")
    X_train_save_path = root / preproc_cfg.get("X_train_path", "data/processed/X_train.csv")
    X_test_save_path = root / preproc_cfg.get("X_test_path", "data/processed/X_test.csv")
    y_train_save_path = root / preproc_cfg.get("y_train_path", "data/processed/y_train.csv")
    y_test_save_path = root / preproc_cfg.get("y_test_path", "data/processed/y_test.csv")

    # 1. Load split datasets
    logger.info(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    logger.info(f"Loading testing data from: {test_path}")
    test_df = pd.read_csv(test_path)

    # 2. Separate features and target
    X_train = train_df[num_cols + cat_cols].copy()
    y_train = train_df[[target_col]].copy()

    X_test = test_df[num_cols + cat_cols].copy()
    y_test = test_df[[target_col]].copy()

    # 3. Build & fit preprocessor ONLY on X_train (Leakage Prevention)
    preprocessor = build_preprocessor(num_cols, cat_cols, scaler_type=scaler_type)
    preprocessor = fit_preprocessor(preprocessor, X_train)

    # 4. Transform train and test
    X_train_processed = transform_data(preprocessor, X_train)
    X_test_processed = transform_data(preprocessor, X_test)

    # 5. Validate dimension parity
    if X_train_processed.shape[1] != X_test_processed.shape[1]:
        err = (
            f"Dimension mismatch between train ({X_train_processed.shape[1]}) "
            f"and test ({X_test_processed.shape[1]}) features!"
        )
        logger.error(err)
        raise ValueError(err)

    feature_names = list(preprocessor.get_feature_names_out())
    logger.info(f"Transformed features ({len(feature_names)}): {feature_names}")
    logger.info(f"X_train shape: {X_train_processed.shape}, X_test shape: {X_test_processed.shape}")

    # 6. Save artifacts
    save_preprocessor(preprocessor, preproc_save_path)

    X_train_save_path.parent.mkdir(parents=True, exist_ok=True)
    X_train_processed.to_csv(X_train_save_path, index=False)
    X_test_processed.to_csv(X_test_save_path, index=False)
    y_train.to_csv(y_train_save_path, index=False)
    y_test.to_csv(y_test_save_path, index=False)

    logger.info(f"Saved processed features to {X_train_save_path} and {X_test_save_path}")
    logger.info(f"Saved processed targets to {y_train_save_path} and {y_test_save_path}")
    logger.info("================ Preprocessing Pipeline Completed Successfully ================")

    return {
        "preprocessor_path": str(preproc_save_path),
        "X_train_shape": X_train_processed.shape,
        "X_test_shape": X_test_processed.shape,
        "feature_names": feature_names,
        "num_features": len(feature_names),
    }
