"""
Unit Tests — Model Training & Inference
Heart Disease MLOps Assignment
BITS Pilani MTech (AMLCSZG523)
"""

import os
import pickle
import sys

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_preprocessing import (  # noqa: E402
    clean_data,
    load_raw_data,
    preprocess_and_split,
)

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "heart_disease.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# --- Fixtures ---

@pytest.fixture
def synthetic_data():
    """Generate synthetic classification data for offline testing."""
    X, y = make_classification(
        n_samples=200, n_features=13, n_informative=8,
        n_redundant=2, random_state=42
    )
    return X, y


@pytest.fixture
def trained_lr(synthetic_data):
    X, y = synthetic_data
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_rf(synthetic_data):
    X, y = synthetic_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


# --- Tests: Logistic Regression ---

class TestLogisticRegression:
    def test_model_trains_without_error(self, synthetic_data):
        X, y = synthetic_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        assert hasattr(model, "coef_")

    def test_predictions_are_binary(self, trained_lr):
        model, X, _ = trained_lr
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_sums_to_one(self, trained_lr):
        model, X, _ = trained_lr
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_in_range(self, trained_lr):
        model, X, _ = trained_lr
        proba = model.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_accuracy_above_threshold(self, trained_lr):
        from sklearn.metrics import accuracy_score
        model, X, y = trained_lr
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        assert acc > 0.5, f"Expected accuracy > 0.5, got {acc:.4f}"

    def test_single_sample_prediction(self, trained_lr):
        model, X, _ = trained_lr
        sample = X[:1]
        pred = model.predict(sample)
        assert len(pred) == 1
        assert pred[0] in {0, 1}


# --- Tests: Random Forest ---

class TestRandomForest:
    def test_model_trains_without_error(self, synthetic_data):
        X, y = synthetic_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        assert hasattr(model, "estimators_")

    def test_predictions_are_binary(self, trained_rf):
        model, X, _ = trained_rf
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_valid(self, trained_rf):
        model, X, _ = trained_rf
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_feature_importances_available(self, trained_rf):
        model, X, _ = trained_rf
        importances = model.feature_importances_
        assert len(importances) == X.shape[1]
        assert all(imp >= 0 for imp in importances)

    def test_accuracy_above_threshold(self, trained_rf):
        from sklearn.metrics import accuracy_score
        model, X, y = trained_rf
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        assert acc > 0.5, f"Expected accuracy > 0.5, got {acc:.4f}"


# --- Tests: Model on Real Data ---

class TestModelOnRealData:
    @pytest.mark.skipif(not os.path.exists(RAW_DATA_PATH), reason="Raw data not available")
    def test_logistic_regression_on_heart_data(self):
        df = clean_data(load_raw_data())
        X_train, X_test, y_train, y_test, _ = preprocess_and_split(df)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, preds)
        assert acc > 0.70, f"Expected acc > 0.70 on heart data, got {acc:.4f}"

    @pytest.mark.skipif(not os.path.exists(RAW_DATA_PATH), reason="Raw data not available")
    def test_random_forest_on_heart_data(self):
        df = clean_data(load_raw_data())
        X_train, X_test, y_train, y_test, _ = preprocess_and_split(df)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, preds)
        assert acc > 0.75, f"Expected acc > 0.75 on heart data, got {acc:.4f}"


# --- Tests: Model Serialization ---

class TestModelSerialization:
    def test_model_can_be_pickled_and_loaded(self, trained_lr, tmp_path):
        model, X, _ = trained_lr
        path = tmp_path / "test_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        with open(path, "rb") as f:
            loaded_model = pickle.load(f)
        preds_original = model.predict(X)
        preds_loaded = loaded_model.predict(X)
        np.testing.assert_array_equal(preds_original, preds_loaded)

    def test_saved_model_predict_proba_matches(self, trained_rf, tmp_path):
        model, X, _ = trained_rf
        path = tmp_path / "test_rf.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        np.testing.assert_allclose(
            model.predict_proba(X),
            loaded.predict_proba(X),
            atol=1e-6
        )
