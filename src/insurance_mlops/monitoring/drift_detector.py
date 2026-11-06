"""Statistical data drift detection for numerical and categorical features."""

from typing import List, Literal, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import chi2_contingency, ks_2samp

from insurance_mlops.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureDriftResult(BaseModel):
    """Structured result for an individual feature drift test."""

    feature_name: str
    feature_type: Literal["numerical", "categorical"]
    test_name: str
    statistic: float
    p_value: float
    drift_detected: bool
    reference_count: int
    current_count: int
    severity: Literal["NO_DRIFT", "MODERATE_DRIFT", "SEVERE_DRIFT"]
    explanation: str


def detect_numerical_drift(
    reference_series: pd.Series,
    current_series: pd.Series,
    feature_name: str,
    p_value_threshold: float = 0.05,
) -> FeatureDriftResult:
    """Detects distribution drift in continuous numerical features using the two-sample Kolmogorov-Smirnov test.

    Args:
        reference_series: Baseline distribution series.
        current_series: Target/production distribution series.
        feature_name: Name of the feature.
        p_value_threshold: Statistical significance alpha (default: 0.05).

    Returns:
        FeatureDriftResult with test statistics, p-value, and severity assessment.
    """
    ref_clean = reference_series.dropna().to_numpy()
    curr_clean = current_series.dropna().to_numpy()

    ref_count = len(ref_clean)
    curr_count = len(curr_clean)

    # Edge Case: empty series
    if ref_count == 0 or curr_count == 0:
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="numerical",
            test_name="Kolmogorov-Smirnov",
            statistic=0.0,
            p_value=1.0,
            drift_detected=False,
            reference_count=ref_count,
            current_count=curr_count,
            severity="NO_DRIFT",
            explanation="Insufficient data points to perform two-sample KS test.",
        )

    # Edge Case: identical constant columns
    if np.all(ref_clean == ref_clean[0]) and np.all(curr_clean == curr_clean[0]):
        if ref_clean[0] == curr_clean[0]:
            return FeatureDriftResult(
                feature_name=feature_name,
                feature_type="numerical",
                test_name="Kolmogorov-Smirnov",
                statistic=0.0,
                p_value=1.0,
                drift_detected=False,
                reference_count=ref_count,
                current_count=curr_count,
                severity="NO_DRIFT",
                explanation="Constant column with identical constant value across both sets.",
            )
        else:
            return FeatureDriftResult(
                feature_name=feature_name,
                feature_type="numerical",
                test_name="Kolmogorov-Smirnov",
                statistic=1.0,
                p_value=0.0,
                drift_detected=True,
                reference_count=ref_count,
                current_count=curr_count,
                severity="SEVERE_DRIFT",
                explanation=f"Constant column shifted value from {ref_clean[0]} to {curr_clean[0]}.",
            )

    # Two-sample KS test
    ks_res = ks_2samp(ref_clean, curr_clean)
    statistic = round(float(ks_res.statistic), 4)
    p_value = round(float(ks_res.pvalue), 6)

    drift_detected = bool(p_value < p_value_threshold)

    # Distinguish statistical significance from practical magnitude
    warning_prefix = ""
    if curr_count < 30:
        warning_prefix = "Small sample size (N < 30); "

    if not drift_detected:
        severity = "NO_DRIFT"
        explanation = f"{warning_prefix}No statistically significant drift (p={p_value:.4f} >= {p_value_threshold})."
    else:
        if statistic < 0.20:
            severity = "MODERATE_DRIFT"
            explanation = (
                f"{warning_prefix}Statistically significant shift detected with "
                f"moderate effect size (KS={statistic:.4f}, p={p_value:.4f})."
            )
        else:
            severity = "SEVERE_DRIFT"
            explanation = (
                f"{warning_prefix}Severe distribution shift detected with "
                f"high effect size (KS={statistic:.4f}, p={p_value:.4f})."
            )

    return FeatureDriftResult(
        feature_name=feature_name,
        feature_type="numerical",
        test_name="Kolmogorov-Smirnov",
        statistic=statistic,
        p_value=p_value,
        drift_detected=drift_detected,
        reference_count=ref_count,
        current_count=curr_count,
        severity=severity,
        explanation=explanation,
    )


def detect_categorical_drift(
    reference_series: pd.Series,
    current_series: pd.Series,
    feature_name: str,
    p_value_threshold: float = 0.05,
) -> FeatureDriftResult:
    """Detects frequency distribution drift in categorical features using the Chi-Square test of independence.

    Args:
        reference_series: Baseline categorical series.
        current_series: Target categorical series.
        feature_name: Name of the feature.
        p_value_threshold: Significance threshold (default: 0.05).

    Returns:
        FeatureDriftResult with Chi2 statistic, p-value, and severity assessment.
    """
    ref_clean = reference_series.dropna().astype(str)
    curr_clean = current_series.dropna().astype(str)

    ref_count = len(ref_clean)
    curr_count = len(curr_clean)

    if ref_count == 0 or curr_count == 0:
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            test_name="Chi-Square Contingency",
            statistic=0.0,
            p_value=1.0,
            drift_detected=False,
            reference_count=ref_count,
            current_count=curr_count,
            severity="NO_DRIFT",
            explanation="Insufficient categorical observations to execute test.",
        )

    # Union of all observed categories across both sets
    all_categories = sorted(list(set(ref_clean.unique()).union(set(curr_clean.unique()))))

    if len(all_categories) <= 1:
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            test_name="Chi-Square Contingency",
            statistic=0.0,
            p_value=1.0,
            drift_detected=False,
            reference_count=ref_count,
            current_count=curr_count,
            severity="NO_DRIFT",
            explanation=f"Only single category observed: '{all_categories[0]}'. No drift possible.",
        )

    # Build 2 x K contingency table
    ref_counts = [int((ref_clean == cat).sum()) for cat in all_categories]
    curr_counts = [int((curr_clean == cat).sum()) for cat in all_categories]
    contingency_table = np.array([ref_counts, curr_counts])

    # Chi2 test
    try:
        chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
        statistic = round(float(chi2_stat), 4)
        p_value = round(float(p_val), 6)
    except Exception as exc:
        logger.warning(f"Chi2 contingency test issue on {feature_name}: {exc}")
        statistic = 0.0
        p_value = 1.0

    drift_detected = bool(p_value < p_value_threshold)

    warning_prefix = ""
    if curr_count < 30:
        warning_prefix = "Small sample size (N < 30); "

    if not drift_detected:
        severity = "NO_DRIFT"
        explanation = f"{warning_prefix}Category frequencies match baseline (p={p_value:.4f} >= {p_value_threshold})."
    else:
        # Check severity based on p-value and relative category shift
        if p_value < 1e-4:
            severity = "SEVERE_DRIFT"
            explanation = (
                f"{warning_prefix}Substantial shift in category proportions "
                f"(Chi2={statistic:.2f}, p={p_value:.6f})."
            )
        else:
            severity = "MODERATE_DRIFT"
            explanation = (
                f"{warning_prefix}Moderate shift in category proportions "
                f"(Chi2={statistic:.2f}, p={p_value:.4f})."
            )

    return FeatureDriftResult(
        feature_name=feature_name,
        feature_type="categorical",
        test_name="Chi-Square Contingency",
        statistic=statistic,
        p_value=p_value,
        drift_detected=drift_detected,
        reference_count=ref_count,
        current_count=curr_count,
        severity=severity,
        explanation=explanation,
    )


class DriftDetector:
    """Dataset-level drift detector running appropriate statistical tests across all features."""

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ):
        self.p_value_threshold = p_value_threshold
        self.numerical_features = numerical_features or ["age", "bmi", "children"]
        self.categorical_features = categorical_features or ["sex", "smoker", "region"]

    def detect_dataset_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> List[FeatureDriftResult]:
        """Runs drift tests across all numerical and categorical features.

        Args:
            reference_df: Baseline DataFrame.
            current_df: Production or candidate DataFrame.

        Returns:
            List of FeatureDriftResult objects.
        """
        results: List[FeatureDriftResult] = []

        # 1. Numerical Drift (KS Test)
        for num_col in self.numerical_features:
            if num_col in reference_df.columns and num_col in current_df.columns:
                res = detect_numerical_drift(
                    reference_series=reference_df[num_col],
                    current_series=current_df[num_col],
                    feature_name=num_col,
                    p_value_threshold=self.p_value_threshold,
                )
                results.append(res)
            else:
                logger.warning(f"Numerical feature '{num_col}' missing from one of the datasets.")

        # 2. Categorical Drift (Chi-Square)
        for cat_col in self.categorical_features:
            if cat_col in reference_df.columns and cat_col in current_df.columns:
                res = detect_categorical_drift(
                    reference_series=reference_df[cat_col],
                    current_series=current_df[cat_col],
                    feature_name=cat_col,
                    p_value_threshold=self.p_value_threshold,
                )
                results.append(res)
            else:
                logger.warning(f"Categorical feature '{cat_col}' missing from one of the datasets.")

        return results
