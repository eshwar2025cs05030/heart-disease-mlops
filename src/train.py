"""
Model Training with MLflow Experiment Tracking
Heart Disease UCI Dataset - MLOps Assignment
BITS Pilani MTech (AMLCSZG523)
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import mlflow
import mlflow.sklearn

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import load_raw_data, clean_data, preprocess_and_split, save_pipeline

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")
EXPERIMENT_NAME = "heart_disease_classification"


def evaluate_model(model, X_test, y_test, model_name: str, plots_dir: str):
    """Compute all metrics and generate evaluation plots."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(["No Disease", "Disease"])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2,
            label=f"ROC Curve (AUC = {metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    roc_path = os.path.join(plots_dir, f"{model_name.lower().replace(' ', '_')}_roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()

    return metrics, cm_path, roc_path


def train_and_log(model, model_name: str, params: dict,
                  X_train, X_test, y_train, y_test, pipeline):
    """Train a model and log everything to MLflow."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    plots_dir = os.path.join(REPORTS_DIR, "model_plots")
    os.makedirs(plots_dir, exist_ok=True)

    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # Train
        model.fit(X_train, y_train)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        mlflow.log_metric("cv_roc_auc_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_roc_auc_std", float(cv_scores.std()))

        # Evaluate
        metrics, cm_path, roc_path = evaluate_model(
            model, X_test, y_test, model_name, plots_dir
        )
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, float(round(value, 4)))

        # Log artifacts
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model",
                                 registered_model_name=model_name.replace(" ", "_"))

        # Save model locally
        model_path = os.path.join(MODEL_DIR, f"{model_name.lower().replace(' ', '_')}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_path, artifact_path="saved_model")

        run_id = mlflow.active_run().info.run_id

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Run ID: {run_id}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"{'='*60}\n")

    return model, metrics, run_id


def main():
    print("Loading and preprocessing data...")
    df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test, pipeline = preprocess_and_split(df)

    # Save the preprocessing pipeline
    pipeline_path = os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl")
    save_pipeline(pipeline, pipeline_path)

    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = {}

    # --- Model 1: Logistic Regression ---
    lr_params = {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": 42,
    }
    lr_model = LogisticRegression(**lr_params)
    lr_model, lr_metrics, lr_run_id = train_and_log(
        lr_model, "Logistic Regression", lr_params,
        X_train, X_test, y_train, y_test, pipeline
    )
    results["Logistic Regression"] = {"metrics": lr_metrics, "run_id": lr_run_id}

    # --- Model 2: Random Forest ---
    rf_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42,
    }
    rf_model = RandomForestClassifier(**rf_params)
    rf_model, rf_metrics, rf_run_id = train_and_log(
        rf_model, "Random Forest", rf_params,
        X_train, X_test, y_train, y_test, pipeline
    )
    results["Random Forest"] = {"metrics": rf_metrics, "run_id": rf_run_id}

    # --- Select best model ---
    best_model_name = max(results, key=lambda m: results[m]["metrics"]["roc_auc"])
    best_model = lr_model if best_model_name == "Logistic Regression" else rf_model
    print(f"\nBest model: {best_model_name} "
          f"(ROC-AUC={results[best_model_name]['metrics']['roc_auc']:.4f})")

    # Save best model as 'best_model.pkl'
    best_path = os.path.join(MODEL_DIR, "best_model.pkl")
    with open(best_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best model saved to: {best_path}")

    # --- Comparison plot ---
    _plot_model_comparison(results)

    print("\nTraining complete. Run `mlflow ui` to view experiment dashboard.")


def _plot_model_comparison(results: dict):
    """Bar chart comparing both models across all metrics."""
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    model_names = list(results.keys())
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    colors = ["#3498db", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [results[name]["metrics"][m] for m in metrics_to_plot]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=color, alpha=0.8, edgecolor="black")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Metric", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Comparison — Heart Disease Classification", fontsize=15, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_to_plot])
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plots_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "model_plots")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Model comparison plot saved to: {path}")


if __name__ == "__main__":
    main()
