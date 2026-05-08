"""
Unit Tests — Data Preprocessing
Heart Disease MLOps Assignment
BITS Pilani MTech (AMLCSZG523)
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_preprocessing import (
    load_raw_data,
    clean_data,
    build_preprocessing_pipeline,
    preprocess_and_split,
    preprocess_single_input,
    COLUMN_NAMES,
    get_feature_columns,
)

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "heart_disease.csv")


# --- Fixtures ---

@pytest.fixture
def sample_raw_df():
    return pd.DataFrame({
        "age":      [63.0, 37.0, 41.0, 56.0, 57.0],
        "sex":      [1.0, 1.0, 0.0, 1.0, 0.0],
        "cp":       [1.0, 3.0, 2.0, 2.0, 4.0],
        "trestbps": [145.0, 130.0, 130.0, 120.0, 120.0],
        "chol":     [233.0, 250.0, 204.0, 236.0, 354.0],
        "fbs":      [1.0, 0.0, 0.0, 0.0, 0.0],
        "restecg":  [2.0, 0.0, 2.0, 0.0, 0.0],
        "thalach":  [150.0, 187.0, 172.0, 178.0, 163.0],
        "exang":    [0.0, 0.0, 0.0, 0.0, 1.0],
        "oldpeak":  [2.3, 3.5, 1.4, 0.8, 0.6],
        "slope":    [3.0, 3.0, 1.0, 1.0, 1.0],
        "ca":       [0.0, 0.0, 0.0, 0.0, 0.0],
        "thal":     [6.0, 3.0, 3.0, 3.0, 3.0],
        "target":   [0, 0, 0, 0, 0],
    })


@pytest.fixture
def sample_df_with_disease():
    df = pd.DataFrame({
        "age":      [63.0, 37.0, 67.0, 56.0, 44.0],
        "sex":      [1.0, 1.0, 1.0, 1.0, 1.0],
        "cp":       [1.0, 3.0, 4.0, 2.0, 2.0],
        "trestbps": [145.0, 130.0, 160.0, 120.0, 120.0],
        "chol":     [233.0, 250.0, 286.0, 236.0, 263.0],
        "fbs":      [1.0, 0.0, 0.0, 0.0, 0.0],
        "restecg":  [2.0, 0.0, 2.0, 0.0, 0.0],
        "thalach":  [150.0, 187.0, 108.0, 178.0, 173.0],
        "exang":    [0.0, 0.0, 1.0, 0.0, 0.0],
        "oldpeak":  [2.3, 3.5, 1.5, 0.8, 0.0],
        "slope":    [3.0, 3.0, 2.0, 1.0, 1.0],
        "ca":       [0.0, 0.0, 3.0, 0.0, 0.0],
        "thal":     [6.0, 3.0, 3.0, 3.0, 7.0],
        "target":   [0, 0, 2, 0, 0],  # one with disease (value > 0)
    })
    return df


# --- Tests: load_raw_data ---

class TestLoadRawData:
    def test_loads_correct_columns(self):
        if not os.path.exists(RAW_DATA_PATH):
            pytest.skip("Raw data not available")
        df = load_raw_data()
        assert list(df.columns) == COLUMN_NAMES

    def test_loads_expected_row_count(self):
        if not os.path.exists(RAW_DATA_PATH):
            pytest.skip("Raw data not available")
        df = load_raw_data()
        assert len(df) == 303

    def test_na_values_parsed(self):
        if not os.path.exists(RAW_DATA_PATH):
            pytest.skip("Raw data not available")
        df = load_raw_data()
        # ca and thal have missing values encoded as '?'
        assert df.isnull().sum().sum() > 0


# --- Tests: clean_data ---

class TestCleanData:
    def test_target_binarized(self, sample_df_with_disease):
        df_clean = clean_data(sample_df_with_disease)
        assert set(df_clean["target"].unique()).issubset({0, 1})

    def test_disease_patients_mapped_to_1(self, sample_df_with_disease):
        df_clean = clean_data(sample_df_with_disease)
        # Original target = 2 should become 1
        assert 1 in df_clean["target"].values

    def test_no_disease_mapped_to_0(self, sample_df_with_disease):
        df_clean = clean_data(sample_df_with_disease)
        assert 0 in df_clean["target"].values

    def test_no_duplicates_after_cleaning(self, sample_raw_df):
        # Add duplicate row
        df_dup = pd.concat([sample_raw_df, sample_raw_df.iloc[[0]]], ignore_index=True)
        df_clean = clean_data(df_dup)
        assert df_clean.duplicated().sum() == 0

    def test_shape_preserved_without_duplicates(self, sample_raw_df):
        df_clean = clean_data(sample_raw_df)
        assert df_clean.shape[0] == len(sample_raw_df)
        assert df_clean.shape[1] == len(COLUMN_NAMES)


# --- Tests: build_preprocessing_pipeline ---

class TestPreprocessingPipeline:
    def test_pipeline_has_correct_steps(self):
        pipeline = build_preprocessing_pipeline()
        step_names = [name for name, _ in pipeline.steps]
        assert "imputer" in step_names
        assert "scaler" in step_names

    def test_pipeline_transforms_data(self, sample_raw_df):
        pipeline = build_preprocessing_pipeline()
        features = get_feature_columns()
        X = sample_raw_df[features]
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape == (len(sample_raw_df), len(features))
        assert not np.isnan(X_transformed).any()

    def test_pipeline_handles_missing_values(self, sample_raw_df):
        sample_raw_df.loc[0, "ca"] = np.nan
        pipeline = build_preprocessing_pipeline()
        features = get_feature_columns()
        X = sample_raw_df[features]
        X_transformed = pipeline.fit_transform(X)
        assert not np.isnan(X_transformed).any()

    def test_scaled_output_near_zero_mean(self, sample_raw_df):
        pipeline = build_preprocessing_pipeline()
        features = get_feature_columns()
        X = sample_raw_df[features]
        X_transformed = pipeline.fit_transform(X)
        # With StandardScaler, mean should be near 0
        means = X_transformed.mean(axis=0)
        assert all(abs(m) < 1e-10 for m in means)


# --- Tests: preprocess_and_split ---

class TestPreprocessAndSplit:
    def test_split_sizes(self):
        if not os.path.exists(RAW_DATA_PATH):
            pytest.skip("Raw data not available")
        df = clean_data(load_raw_data())
        X_train, X_test, y_train, y_test, _ = preprocess_and_split(df, test_size=0.2)
        total = len(X_train) + len(X_test)
        assert total == len(df)
        assert abs(len(X_test) / total - 0.2) < 0.05

    def test_no_nan_in_output(self):
        if not os.path.exists(RAW_DATA_PATH):
            pytest.skip("Raw data not available")
        df = clean_data(load_raw_data())
        X_train, X_test, y_train, y_test, _ = preprocess_and_split(df)
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

    def test_labels_are_binary(self):
        if not os.path.exists(RAW_DATA_PATH):
            pytest.skip("Raw data not available")
        df = clean_data(load_raw_data())
        _, _, y_train, y_test, _ = preprocess_and_split(df)
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})


# --- Tests: preprocess_single_input ---

class TestSingleInputPreprocessing:
    def test_single_input_returns_2d_array(self):
        if not os.path.exists(RAW_DATA_PATH):
            pytest.skip("Raw data not available")
        df = clean_data(load_raw_data())
        features = get_feature_columns()
        X = df[features]
        pipeline = build_preprocessing_pipeline()
        pipeline.fit(X)

        sample_input = {
            "age": 54.0, "sex": 1.0, "cp": 2.0, "trestbps": 130.0,
            "chol": 250.0, "fbs": 0.0, "restecg": 0.0, "thalach": 150.0,
            "exang": 0.0, "oldpeak": 1.5, "slope": 2.0, "ca": 0.0, "thal": 3.0
        }
        result = preprocess_single_input(sample_input, pipeline)
        assert result.ndim == 2
        assert result.shape == (1, len(features))
        assert not np.isnan(result).any()
