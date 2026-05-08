"""
Data Preprocessing Module
Heart Disease UCI Dataset - MLOps Assignment
BITS Pilani MTech (AMLCSZG523)
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "heart_disease.csv")
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw UCI Heart Disease dataset."""
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES, na_values="?")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Replace '?' with NaN (already done at load time)
    - Binarize target (0 = no disease, 1 = disease)
    - Drop duplicates
    """
    df = df.copy()
    # Binarize target: original has values 0-4; 0 = no disease, 1-4 = disease
    df["target"] = (df["target"] > 0).astype(int)
    df = df.drop_duplicates()
    return df


def get_feature_columns():
    return [c for c in COLUMN_NAMES if c != "target"]


def build_preprocessing_pipeline() -> Pipeline:
    """
    Build a sklearn Pipeline for preprocessing:
    - Impute missing values with median
    - Standard scale all features
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return pipeline


def preprocess_and_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing: split into train/test and fit the pipeline on train.
    Returns X_train, X_test, y_train, y_test, fitted_pipeline
    """
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_preprocessing_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, pipeline


def save_pipeline(pipeline: Pipeline, path: str):
    """Save the fitted preprocessing pipeline."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved to {path}")


def load_pipeline(path: str) -> Pipeline:
    """Load a saved preprocessing pipeline."""
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


def preprocess_single_input(data: dict, pipeline: Pipeline) -> np.ndarray:
    """
    Preprocess a single input dict (for API inference).
    Returns a 2D numpy array ready for model prediction.
    """
    feature_cols = get_feature_columns()
    row = pd.DataFrame([data])[feature_cols]
    return pipeline.transform(row)


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Raw data shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    df_clean = clean_data(df)
    print(f"Clean data shape: {df_clean.shape}")
    print(f"Class distribution:\n{df_clean['target'].value_counts()}")
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    df_clean.to_csv(os.path.join(PROCESSED_DATA_PATH, "heart_disease_clean.csv"), index=False)
    print("Clean data saved.")
