"""Unit tests for statistical data drift detection algorithms."""

import numpy as np
import pandas as pd

from insurance_mlops.monitoring.drift_detector import (
    DriftDetector,
    detect_categorical_drift,
    detect_numerical_drift,
)


def test_numerical_drift_identical():
    """Tests that identical distributions show no drift (high p-value)."""
    np.random.seed(42)
    ref = pd.Series(np.random.normal(30.0, 5.0, 200), name="age")
    curr = pd.Series(np.random.normal(30.0, 5.0, 200), name="age")

    result = detect_numerical_drift(ref, curr, "age", p_value_threshold=0.05)

    assert result.drift_detected is False
    assert result.p_value > 0.05
    assert result.severity == "NO_DRIFT"


def test_numerical_drift_shifted():
    """Tests that a significantly shifted distribution is flagged as severe drift."""
    np.random.seed(42)
    ref = pd.Series(np.random.normal(30.0, 5.0, 200), name="age")
    # Deliberate severe shift: mean shifted from 30 to 55
    curr = pd.Series(np.random.normal(55.0, 5.0, 200), name="age")

    result = detect_numerical_drift(ref, curr, "age", p_value_threshold=0.05)

    assert result.drift_detected is True
    assert result.p_value < 1e-4
    assert result.statistic > 0.5
    assert result.severity == "SEVERE_DRIFT"


def test_categorical_drift_identical():
    """Tests that identical categorical frequencies detect no drift."""
    ref = pd.Series(["no"] * 80 + ["yes"] * 20, name="smoker")
    curr = pd.Series(["no"] * 80 + ["yes"] * 20, name="smoker")

    result = detect_categorical_drift(ref, curr, "smoker", p_value_threshold=0.05)

    assert result.drift_detected is False
    assert result.p_value > 0.05
    assert result.severity == "NO_DRIFT"


def test_categorical_drift_shifted():
    """Tests that inverted categorical proportions trigger drift alert."""
    ref = pd.Series(["no"] * 80 + ["yes"] * 20, name="smoker")
    # Drastic shift: smokers jump from 20% to 80%
    curr = pd.Series(["no"] * 20 + ["yes"] * 80, name="smoker")

    result = detect_categorical_drift(ref, curr, "smoker", p_value_threshold=0.05)

    assert result.drift_detected is True
    assert result.p_value < 1e-5
    assert result.severity == "SEVERE_DRIFT"


def test_constant_columns_handling():
    """Tests that constant numerical and categorical columns are handled safely without crashing."""
    # Identical constant numeric columns
    ref_const = pd.Series([10.0] * 50)
    curr_const = pd.Series([10.0] * 50)
    res_num = detect_numerical_drift(ref_const, curr_const, "const_num")
    assert res_num.drift_detected is False

    # Shifted constant numeric columns
    curr_const_diff = pd.Series([20.0] * 50)
    res_num_diff = detect_numerical_drift(ref_const, curr_const_diff, "const_num")
    assert res_num_diff.drift_detected is True

    # Single category column
    ref_cat = pd.Series(["same"] * 50)
    curr_cat = pd.Series(["same"] * 50)
    res_cat = detect_categorical_drift(ref_cat, curr_cat, "const_cat")
    assert res_cat.drift_detected is False


def test_small_sample_handling():
    """Tests that very small sample sizes (N < 30) include appropriate warning in explanation."""
    ref = pd.Series([20, 25, 30, 35, 40] * 10)
    curr = pd.Series([22, 28, 33])  # N=3

    res = detect_numerical_drift(ref, curr, "age")
    assert "Small sample size" in res.explanation
    assert res.current_count == 3


def test_detector_dataset_evaluation():
    """Tests multi-feature dataset evaluation."""
    ref_df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "bmi": [22.0, 26.0, 30.0, 34.0],
        "children": [0, 1, 2, 3],
        "sex": ["male", "female", "male", "female"],
        "smoker": ["no", "no", "yes", "no"],
        "region": ["northeast", "northwest", "southeast", "southwest"],
    })
    curr_df = ref_df.copy()

    detector = DriftDetector()
    results = detector.detect_dataset_drift(ref_df, curr_df)

    assert len(results) == 6
    assert all(not r.drift_detected for r in results)
