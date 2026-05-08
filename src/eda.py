"""
Exploratory Data Analysis (EDA) Module
Heart Disease UCI Dataset - MLOps Assignment
BITS Pilani MTech (AMLCSZG523)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CI/CD environments
import matplotlib.pyplot as plt
import seaborn as sns
import os

from data_preprocessing import load_raw_data, clean_data, COLUMN_NAMES

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "eda_plots")

FEATURE_DESCRIPTIONS = {
    "age": "Age (years)",
    "sex": "Sex (1=Male, 0=Female)",
    "cp": "Chest Pain Type (1-4)",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl",
    "restecg": "Resting ECG Results",
    "thalach": "Max Heart Rate Achieved",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression (Exercise vs Rest)",
    "slope": "Slope of Peak Exercise ST",
    "ca": "Major Vessels (0-3) by Fluoroscopy",
    "thal": "Thalassemia",
    "target": "Heart Disease (1=Yes, 0=No)",
}


def plot_class_distribution(df: pd.DataFrame, save_dir: str):
    """Bar chart of target class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2ecc71", "#e74c3c"]

    counts = df["target"].value_counts().sort_index()
    labels = ["No Disease (0)", "Heart Disease (1)"]

    axes[0].bar(labels, counts.values, color=colors, edgecolor="black", width=0.5)
    axes[0].set_title("Class Distribution (Count)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Number of Patients")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 2, str(v), ha="center", fontweight="bold")

    axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=140, wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Class Distribution (Proportion)", fontsize=14, fontweight="bold")

    plt.suptitle("Heart Disease Target Class Balance", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_feature_histograms(df: pd.DataFrame, save_dir: str):
    """Histogram for each feature, coloured by target."""
    feature_cols = [c for c in COLUMN_NAMES if c != "target"]
    n_cols = 4
    n_rows = int(np.ceil(len(feature_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()

    palette = {0: "#2ecc71", 1: "#e74c3c"}
    for i, col in enumerate(feature_cols):
        for target_val, color in palette.items():
            subset = df[df["target"] == target_val][col].dropna()
            axes[i].hist(subset, bins=20, alpha=0.6, color=color,
                         label=f"{'Disease' if target_val else 'No Disease'}", edgecolor="white")
        axes[i].set_title(FEATURE_DESCRIPTIONS.get(col, col), fontsize=10, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Heart Disease Status", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "feature_histograms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str):
    """Pearson correlation heatmap of all features including target."""
    fig, ax = plt.subplots(figsize=(14, 11))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
        annot=True, fmt=".2f", linewidths=0.5, ax=ax,
        annot_kws={"size": 9}, square=True,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(save_dir, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_boxplots(df: pd.DataFrame, save_dir: str):
    """Boxplots of continuous features grouped by target."""
    continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(1, len(continuous_cols), figsize=(20, 6))
    palette = {0: "#2ecc71", 1: "#e74c3c"}

    for i, col in enumerate(continuous_cols):
        data_by_class = [
            df[df["target"] == 0][col].dropna().values,
            df[df["target"] == 1][col].dropna().values,
        ]
        bp = axes[i].boxplot(data_by_class, patch_artist=True,
                             labels=["No Disease", "Disease"])
        for patch, color in zip(bp["boxes"], ["#2ecc71", "#e74c3c"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[i].set_title(FEATURE_DESCRIPTIONS.get(col, col), fontsize=11, fontweight="bold")
        axes[i].set_ylabel(col)

    plt.suptitle("Continuous Feature Distribution by Heart Disease Status",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def run_full_eda(df: pd.DataFrame = None, save_dir: str = OUTPUT_DIR):
    """Run the full EDA pipeline and save all plots."""
    os.makedirs(save_dir, exist_ok=True)
    if df is None:
        df = clean_data(load_raw_data())

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS - HEART DISEASE DATASET")
    print("=" * 60)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nBasic statistics:\n{df.describe().round(2)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nClass distribution:\n{df['target'].value_counts()}")

    paths = []
    paths.append(plot_class_distribution(df, save_dir))
    paths.append(plot_feature_histograms(df, save_dir))
    paths.append(plot_correlation_heatmap(df, save_dir))
    paths.append(plot_boxplots(df, save_dir))

    print(f"\nAll EDA plots saved to: {save_dir}")
    return paths


if __name__ == "__main__":
    run_full_eda()
