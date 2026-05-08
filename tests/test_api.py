"""
Unit Tests — FastAPI Application
Heart Disease MLOps Assignment
BITS Pilani MTech (AMLCSZG523)
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


VALID_PATIENT = {
    "age": 54.0, "sex": 1.0, "cp": 2.0, "trestbps": 130.0,
    "chol": 250.0, "fbs": 0.0, "restecg": 0.0, "thalach": 150.0,
    "exang": 0.0, "oldpeak": 1.5, "slope": 2.0, "ca": 0.0, "thal": 3.0
}

HIGH_RISK_PATIENT = {
    "age": 67.0, "sex": 1.0, "cp": 4.0, "trestbps": 160.0,
    "chol": 286.0, "fbs": 0.0, "restecg": 2.0, "thalach": 108.0,
    "exang": 1.0, "oldpeak": 1.5, "slope": 2.0, "ca": 3.0, "thal": 3.0
}


def make_mock_pipeline():
    """Create a real sklearn pipeline trained on synthetic data."""
    X, _ = make_classification(n_samples=100, n_features=13, random_state=42)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pipeline.fit(X)
    return pipeline


def make_mock_model(pipeline):
    """Create a real LR model trained on synthetic data."""
    X, y = make_classification(n_samples=100, n_features=13, random_state=42)
    X_proc = pipeline.transform(X)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_proc, y)
    return model


@pytest.fixture(scope="module")
def client():
    """TestClient with mocked model and pipeline."""
    mock_pipeline = make_mock_pipeline()
    mock_model = make_mock_model(mock_pipeline)

    import app as app_module
    app_module.model = mock_model
    app_module.pipeline = mock_pipeline

    from app import app
    return TestClient(app)


# --- Tests: Health endpoints ---

class TestHealthEndpoints:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_message(self, client):
        response = client.get("/")
        data = response.json()
        assert "message" in data

    def test_health_endpoint_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_shows_model_loaded(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True
        assert data["pipeline_loaded"] is True


# --- Tests: /predict endpoint ---

class TestPredictEndpoint:
    def test_valid_input_returns_200(self, client):
        response = client.post("/predict", json=VALID_PATIENT)
        assert response.status_code == 200

    def test_prediction_is_binary(self, client):
        response = client.post("/predict", json=VALID_PATIENT)
        data = response.json()
        assert data["prediction"] in [0, 1]

    def test_probabilities_sum_to_one(self, client):
        response = client.post("/predict", json=VALID_PATIENT)
        data = response.json()
        total = data["probability_no_disease"] + data["probability_disease"]
        assert abs(total - 1.0) < 0.01

    def test_confidence_between_0_and_1(self, client):
        response = client.post("/predict", json=VALID_PATIENT)
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_prediction_label_matches_prediction(self, client):
        response = client.post("/predict", json=VALID_PATIENT)
        data = response.json()
        if data["prediction"] == 1:
            assert "Heart Disease" in data["prediction_label"]
        else:
            assert "No Heart Disease" in data["prediction_label"]

    def test_response_has_all_fields(self, client):
        response = client.post("/predict", json=VALID_PATIENT)
        data = response.json()
        required_fields = [
            "prediction", "prediction_label", "confidence",
            "probability_no_disease", "probability_disease", "model_version"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_missing_field_returns_422(self, client):
        incomplete = {k: v for k, v in VALID_PATIENT.items() if k != "age"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_invalid_age_returns_422(self, client):
        invalid = VALID_PATIENT.copy()
        invalid["age"] = -5.0
        response = client.post("/predict", json=invalid)
        assert response.status_code == 422

    def test_invalid_sex_returns_422(self, client):
        invalid = VALID_PATIENT.copy()
        invalid["sex"] = 5.0
        response = client.post("/predict", json=invalid)
        assert response.status_code == 422

    def test_both_patients_return_valid_predictions(self, client):
        for patient in [VALID_PATIENT, HIGH_RISK_PATIENT]:
            response = client.post("/predict", json=patient)
            assert response.status_code == 200
            data = response.json()
            assert data["prediction"] in [0, 1]
