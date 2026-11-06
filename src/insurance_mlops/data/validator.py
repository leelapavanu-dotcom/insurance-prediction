"""Data schema validation and quality verification using Pydantic."""

from typing import Dict, List, Literal, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, Field

from insurance_mlops.utils.logger import get_logger

logger = get_logger(__name__)


class InsuranceRecord(BaseModel):
    """Pydantic model representing a single insurance dataset observation."""

    age: int = Field(..., ge=18, le=100, description="Age of primary beneficiary (18-100)")
    sex: Literal["female", "male"] = Field(..., description="Gender of primary beneficiary")
    bmi: float = Field(..., ge=10.0, le=70.0, description="Body mass index (10.0-70.0)")
    children: int = Field(..., ge=0, le=20, description="Number of children/dependents covered (0-20)")
    smoker: Literal["yes", "no"] = Field(..., description="Smoking status")
    region: Literal["northeast", "northwest", "southeast", "southwest"] = Field(
        ..., description="Residential region in the US"
    )
    charges: Optional[float] = Field(
        default=None, ge=0.0, description="Individual medical costs billed by health insurance"
    )


class ValidationReport(BaseModel):
    """Validation report summarizing data quality checks."""

    total_rows: int
    valid_rows: int
    duplicate_count: int
    dropped_duplicates: int
    missing_values: Dict[str, int]
    is_valid: bool
    details: List[str] = []


def validate_schema(
    df: pd.DataFrame,
    is_training: bool = True,
    drop_duplicates: bool = False,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """Validates an insurance DataFrame against Pydantic schema and quality rules.

    Args:
        df: Input DataFrame to validate.
        is_training: If True, requires the target column 'charges' to be present.
        drop_duplicates: If True, drops duplicate rows; otherwise retains them with a report.

    Returns:
        Tuple of (validated_df, ValidationReport).

    Raises:
        ValueError: If essential columns are missing or critical schema violations occur.
    """
    logger.info(f"Starting data validation (is_training={is_training}, drop_duplicates={drop_duplicates})...")
    details: List[str] = []
    initial_row_count = len(df)

    # 1. Required Column Check
    expected_features = ["age", "sex", "bmi", "children", "smoker", "region"]
    required_columns = expected_features + (["charges"] if is_training else [])
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        err = f"Validation Error: Missing required columns: {missing_cols}"
        logger.error(err)
        raise ValueError(err)

    # 2. Missing Value Check
    missing_counts = df[required_columns].isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())
    if total_missing > 0:
        err = f"Validation Error: Found {total_missing} missing values: {missing_counts}"
        logger.error(err)
        raise ValueError(err)
    details.append("Zero missing values detected.")

    # 3. Duplicate Row Detection & Handling
    # Check duplicates across feature columns (or all columns if target present)
    dup_mask = df.duplicated(keep="first")
    dup_count = int(dup_mask.sum())

    working_df = df.copy()
    dropped_count = 0

    if dup_count > 0:
        dup_rows = df[dup_mask]
        logger.warning(
            f"Detected {dup_count} duplicate row(s) in dataset (indices: {dup_rows.index.tolist()})."
        )
        if drop_duplicates:
            working_df = working_df.drop_duplicates(keep="first").reset_index(drop=True)
            dropped_count = dup_count
            msg = f"Configured to drop duplicates: removed {dup_count} duplicate row(s)."
            logger.info(msg)
            details.append(msg)
        else:
            msg = f"Configured to retain duplicates: {dup_count} duplicate row(s) preserved."
            logger.info(msg)
            details.append(msg)
    else:
        details.append("No duplicate rows detected.")

    # 4. Row-Level Schema Validation via Pydantic
    errors: List[str] = []
    # Fast vectorized check for domain boundaries
    invalid_mask = (
        (working_df["age"] < 18)
        | (working_df["age"] > 100)
        | (~working_df["sex"].isin(["female", "male"]))
        | (working_df["bmi"] < 10.0)
        | (working_df["bmi"] > 70.0)
        | (working_df["children"] < 0)
        | (working_df["children"] > 20)
        | (~working_df["smoker"].isin(["yes", "no"]))
        | (~working_df["region"].isin(["northeast", "northwest", "southeast", "southwest"]))
    )
    if is_training and "charges" in working_df.columns:
        invalid_mask = invalid_mask | (working_df["charges"] < 0)

    if invalid_mask.any():
        bad_indices = working_df[invalid_mask].index.tolist()
        err_msg = f"Found {len(bad_indices)} row(s) violating schema constraints at indices: {bad_indices[:10]}"
        logger.error(err_msg)
        errors.append(err_msg)
        raise ValueError(err_msg)

    details.append(f"All {len(working_df)} rows conform strictly to Pydantic schema specifications.")

    report = ValidationReport(
        total_rows=initial_row_count,
        valid_rows=len(working_df),
        duplicate_count=dup_count,
        dropped_duplicates=dropped_count,
        missing_values=missing_counts,
        is_valid=True,
        details=details,
    )

    logger.info(
        f"Data validation passed: {report.valid_rows}/{report.total_rows} rows valid, "
        f"{report.duplicate_count} duplicates reported."
    )
    return working_df, report
