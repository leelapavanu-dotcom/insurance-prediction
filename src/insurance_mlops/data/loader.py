"""Data loading, ingestion, splitting, and persistence pipeline."""

from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from insurance_mlops.data.validator import ValidationReport, validate_schema
from insurance_mlops.utils.common import get_project_root, load_yaml
from insurance_mlops.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Loads the raw insurance CSV dataset from disk.

    Args:
        file_path: Relative or absolute path to the raw dataset. If None,
                   loads the path specified in config/config.yaml.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if file_path is None:
        config = load_yaml("config/config.yaml")
        file_path = config["data"]["raw_path"]

    path = Path(file_path)
    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        err = f"Raw dataset not found at: {path}"
        logger.error(err)
        raise FileNotFoundError(err)

    logger.info(f"Loading raw dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded raw dataset successfully. Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a dataset into training and testing partitions.

    Args:
        df: Input DataFrame to partition.
        test_size: Proportion of dataset allocated to test split (e.g. 0.2 for 80/20).
        random_state: Random seed for reproducibility.
        stratify_by: Optional categorical column to stratify splits by (e.g. 'smoker').

    Returns:
        Tuple of (train_df, test_df).
    """
    logger.info(
        f"Splitting dataset: test_size={test_size}, random_state={random_state}, stratify_by={stratify_by}"
    )

    stratify_col = df[stratify_by] if (stratify_by and stratify_by in df.columns) else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(f"Dataset split complete: Train={train_df.shape[0]} rows, Test={test_df.shape[0]} rows.")
    return train_df, test_df


def save_split_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processed_dir: Optional[str] = None,
    reference_path: Optional[str] = None,
) -> Dict[str, str]:
    """Saves train, test, and baseline reference datasets to disk.

    Args:
        train_df: Training DataFrame.
        test_df: Testing DataFrame.
        processed_dir: Directory where processed train.csv and test.csv will be saved.
        reference_path: File path where reference baseline data for drift detection will be stored.

    Returns:
        Dictionary containing paths to the saved files.
    """
    root = get_project_root()
    config = load_yaml("config/config.yaml")

    proc_dir = Path(processed_dir or config["data"]["processed_dir"])
    if not proc_dir.is_absolute():
        proc_dir = root / proc_dir
    proc_dir.mkdir(parents=True, exist_ok=True)

    ref_file = Path(reference_path or config["data"]["reference_path"])
    if not ref_file.is_absolute():
        ref_file = root / ref_file
    ref_file.parent.mkdir(parents=True, exist_ok=True)

    train_path = proc_dir / "train.csv"
    test_path = proc_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    # The reference dataset acts as the baseline for data drift monitoring
    train_df.to_csv(ref_file, index=False)

    logger.info(f"Saved training dataset to: {train_path} ({len(train_df)} rows)")
    logger.info(f"Saved testing dataset to: {test_path} ({len(test_df)} rows)")
    logger.info(f"Saved baseline reference dataset to: {ref_file} ({len(train_df)} rows)")

    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "reference_path": str(ref_file),
    }


def run_data_pipeline(
    config_path: str = "config/config.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame, ValidationReport]:
    """Executes the complete data ingestion and validation pipeline.

    1. Loads project configuration.
    2. Ingests raw insurance data.
    3. Executes Pydantic schema validation and configurable duplicate handling.
    4. Splits data into train and test sets.
    5. Persists train, test, and reference datasets.

    Args:
        config_path: Path to configuration YAML.

    Returns:
        Tuple of (train_df, test_df, validation_report).
    """
    logger.info("================ Starting Data Ingestion & Validation Pipeline ================")
    config = load_yaml(config_path)

    raw_path = config["data"]["raw_path"]
    drop_dups = config["data"].get("drop_duplicates", False)
    test_size = config["data"].get("test_size", 0.2)
    random_seed = config["project"].get("random_seed", 42)
    stratify_by = config["data"].get("stratify_by", None)

    # Ingest
    df_raw = load_raw_data(raw_path)

    # Validate
    df_validated, report = validate_schema(
        df_raw,
        is_training=True,
        drop_duplicates=drop_dups,
    )

    # Split
    train_df, test_df = split_data(
        df_validated,
        test_size=test_size,
        random_state=random_seed,
        stratify_by=stratify_by,
    )

    # Persist
    save_split_data(
        train_df,
        test_df,
        processed_dir=config["data"]["processed_dir"],
        reference_path=config["data"]["reference_path"],
    )

    logger.info("================ Data Ingestion Pipeline Completed Successfully ================")
    return train_df, test_df, report


if __name__ == "__main__":
    run_data_pipeline()
