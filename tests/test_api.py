"""Integration tests for FastAPI inference endpoints."""

import pytest
from fastapi.testclient import TestClient

from insurance_mlops.inference.api import app


@pytest.fixture(scope="module")
def client():
    """Initializes FastAPI TestClient for endpoint integration tests."""
    with TestClient(app) as test_client:
        yield test_client


def test_api_root(client):
    """Tests root info endpoint returns 200 with documentation URLs."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "/docs" in data["docs_url"]


def test_api_health(client):
    """Tests health endpoint reports healthy with champion model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["artifacts_loaded"] is True
    assert data["model_name"] in ["random_forest", "gradient_boosting", "linear_regression"]


def test_api_predict_single_record(client):
    """Tests single patient prediction request."""
    payload = {
        "age": 30,
        "sex": "female",
        "bmi": 28.5,
        "children": 1,
        "smoker": "no",
        "region": "northwest",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_charges" in data
    assert data["predicted_charges"] > 1000.0
    assert data["model_name"] != ""


def test_api_predict_batch_records(client):
    """Tests batch patient prediction request."""
    payload = [
        {"age": 20, "sex": "male", "bmi": 25.0, "children": 0, "smoker": "no", "region": "southwest"},
        {"age": 55, "sex": "female", "bmi": 35.0, "children": 2, "smoker": "yes", "region": "northeast"},
    ]
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    # Smoker charges should exceed non-smoker charges
    assert data["predictions"][1] > data["predictions"][0]


def test_api_predict_invalid_schema(client):
    """Tests that payload violating domain rules returns 422 Unprocessable Entity."""
    # Invalid underage age (age = 15)
    invalid_payload = {
        "age": 15,
        "sex": "female",
        "bmi": 25.0,
        "children": 0,
        "smoker": "no",
        "region": "southwest",
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_api_predict_missing_required_fields(client):
    """Tests that missing required fields returns 422 Unprocessable Entity."""
    incomplete_payload = {
        "age": 40,
        "sex": "male",
        # Missing bmi, children, smoker, region
    }
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422
