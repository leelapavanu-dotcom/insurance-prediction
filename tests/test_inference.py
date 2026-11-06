"""Unit and integration tests for the inference engine."""

from pathlib import Path
import pandas as pd
from pydantic import ValidationError
import pytest

from insurance_mlops.inference.predictor import (
    InsurancePredictor,
    predict_batch_csv,
)


@pytest.fixture
def predictor():
    """Initializes the production InsurancePredictor with local artifacts."""
    return InsurancePredictor()


def test_predictor_initialization(predictor):
    """Verifies that model and preprocessor artifacts are loaded cleanly."""
    assert predictor.model is not None
    assert predictor.preprocessor is not None
    assert predictor.model_name in ["random_forest", "gradient_boosting", "linear_regression"]


def test_predict_single_record(predictor):
    """Tests single observation prediction for reasonable numerical values."""
    record = {
        "age": 30,
        "sex": "female",
        "bmi": 26.5,
        "children": 1,
        "smoker": "no",
        "region": "northwest",
    }
    charge = predictor.predict_single(record)

    assert isinstance(charge, float)
    assert charge > 1000.0
    assert charge < 40000.0


def test_predict_smoker_impact(predictor):
    """Verifies that smoker charges are substantially higher than non-smoker charges."""
    non_smoker = {"age": 40, "sex": "male", "bmi": 30.0, "children": 0, "smoker": "no", "region": "southeast"}
    smoker = {"age": 40, "sex": "male", "bmi": 30.0, "children": 0, "smoker": "yes", "region": "southeast"}

    charge_ns = predictor.predict_single(non_smoker)
    charge_s = predictor.predict_single(smoker)

    assert charge_s > charge_ns * 2.0, f"Smoker (${charge_s}) should be > 2x non-smoker (${charge_ns})"


def test_predict_batch_records(predictor):
    """Tests batch prediction across multiple records."""
    records = [
        {"age": 19, "sex": "female", "bmi": 27.9, "children": 0, "smoker": "yes", "region": "southwest"},
        {"age": 18, "sex": "male", "bmi": 33.7, "children": 1, "smoker": "no", "region": "southeast"},
    ]
    charges = predictor.predict_records(records)

    assert len(charges) == 2
    assert charges[0] > charges[1]  # 19-yr old smoker vs 18-yr old non-smoker


def test_invalid_input_validation(predictor):
    """Tests that schema violations raise ValidationError."""
    # Underage patient (age < 18)
    with pytest.raises(ValidationError):
        predictor.predict_single({
            "age": 15,
            "sex": "female",
            "bmi": 22.0,
            "children": 0,
            "smoker": "no",
            "region": "northeast",
        })

    # Unrecognized smoker status
    with pytest.raises(ValidationError):
        predictor.predict_single({
            "age": 25,
            "sex": "female",
            "bmi": 22.0,
            "children": 0,
            "smoker": "sometimes",
            "region": "northeast",
        })


def test_missing_fields_validation(predictor):
    """Tests that missing required fields raise ValidationError."""
    with pytest.raises(ValidationError):
        predictor.predict_single({
            "age": 35,
            "sex": "male",
            # missing bmi, children, smoker, region
        })


def test_predict_dataframe(predictor):
    """Tests DataFrame prediction appends 'predicted_charges'."""
    df = pd.DataFrame([
        {"age": 25, "sex": "female", "bmi": 24.0, "children": 0, "smoker": "no", "region": "southwest"},
        {"age": 55, "sex": "male", "bmi": 32.0, "children": 2, "smoker": "yes", "region": "northeast"},
    ])
    result_df = predictor.predict_dataframe(df)

    assert "predicted_charges" in result_df.columns
    assert len(result_df) == 2
    assert not result_df["predicted_charges"].isnull().any()


def test_predict_batch_csv(tmp_path, predictor):
    """Tests offline CSV batch prediction."""
    in_csv = tmp_path / "input.csv"
    out_csv = tmp_path / "output.csv"

    pd.DataFrame([
        {"age": 22, "sex": "male", "bmi": 28.0, "children": 0, "smoker": "no", "region": "northwest"},
        {"age": 45, "sex": "female", "bmi": 35.0, "children": 3, "smoker": "yes", "region": "southeast"},
    ]).to_csv(in_csv, index=False)

    saved_path = predict_batch_csv(in_csv, out_csv, predictor=predictor)

    assert Path(saved_path).exists()
    df_out = pd.read_csv(saved_path)
    assert len(df_out) == 2
    assert "predicted_charges" in df_out.columns
