"""Professional EDA and model evaluation visualizations."""

import os
import tempfile
from pathlib import Path
from typing import Dict

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "credit_scoring_mpl_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import (
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    SCREENSHOTS_DIR,
    TARGET_COLUMN,
    TARGET_LABEL_COLUMN,
)


sns.set_theme(style="whitegrid", context="notebook")


def _prepare_output_dir() -> None:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_current_figure(filename: str) -> None:
    _prepare_output_dir()
    output_path = SCREENSHOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Save the target class distribution chart."""
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        data=df,
        x=TARGET_LABEL_COLUMN,
        hue=TARGET_LABEL_COLUMN,
        palette=["#d95f59", "#2b8cbe"],
        legend=False,
    )
    total = len(df)
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{int(value.get_height())}\n{value.get_height() / total:.1%}" for value in container],
            fontsize=10,
        )
    ax.set_title("Creditworthiness Class Distribution", fontsize=15, weight="bold")
    ax.set_xlabel("Credit Status")
    ax.set_ylabel("Number of Applicants")
    _save_current_figure("class_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Save a correlation heatmap for numeric features and the target."""
    columns = NUMERIC_FEATURES + [TARGET_COLUMN]
    correlation = df[columns].corr(numeric_only=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        correlation,
        annot=True,
        cmap="vlag",
        center=0,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.82},
    )
    plt.title("Correlation Heatmap", fontsize=15, weight="bold")
    _save_current_figure("correlation_heatmap.png")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """Save histograms for key numeric features."""
    features = [
        "age",
        "income",
        "loan_amount",
        "debt",
        "existing_loans",
        "loan_duration_months",
        "debt_to_income_ratio",
        "loan_to_income_ratio",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for feature, ax in zip(features, axes.flatten()):
        sns.histplot(
            data=df,
            x=feature,
            hue=TARGET_LABEL_COLUMN,
            kde=True,
            bins=24,
            palette=["#d95f59", "#2b8cbe"],
            alpha=0.55,
            ax=ax,
        )
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("")
    fig.suptitle("Feature Distributions by Credit Status", fontsize=16, weight="bold")
    _save_current_figure("feature_distributions.png")


def plot_boxplots(df: pd.DataFrame) -> None:
    """Save boxplots for numeric features against credit status."""
    features = ["age", "income", "loan_amount", "debt"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for feature, ax in zip(features, axes.flatten()):
        sns.boxplot(
            data=df,
            x=TARGET_LABEL_COLUMN,
            y=feature,
            hue=TARGET_LABEL_COLUMN,
            palette=["#d95f59", "#2b8cbe"],
            legend=False,
            ax=ax,
        )
        ax.set_title(f"{feature.replace('_', ' ').title()} by Credit Status")
        ax.set_xlabel("")
    fig.suptitle("Boxplots of Financial Features", fontsize=16, weight="bold")
    _save_current_figure("boxplots_by_credit_status.png")


def plot_pairplot(df: pd.DataFrame) -> None:
    """Save a pairplot for a focused subset of features."""
    columns = ["age", "income", "loan_amount", "debt", TARGET_LABEL_COLUMN]
    pair_grid = sns.pairplot(
        df[columns],
        hue=TARGET_LABEL_COLUMN,
        corner=True,
        diag_kind="hist",
        palette=["#d95f59", "#2b8cbe"],
        plot_kws={"alpha": 0.65, "s": 28},
    )
    pair_grid.fig.suptitle("Pairplot of Key Credit Features", y=1.02, fontsize=15)
    _prepare_output_dir()
    pair_grid.savefig(SCREENSHOTS_DIR / "pairplot_key_features.png", dpi=180)
    plt.close("all")


def plot_confusion_matrix(model_name: str, confusion_matrix_values) -> None:
    """Save a confusion matrix plot for one model."""
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix_values,
        display_labels=["Bad Credit", "Good Credit"],
    )
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    display.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, weight="bold")
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    _save_current_figure(filename)


def plot_model_comparison(metrics_df: pd.DataFrame) -> None:
    """Save a side-by-side model metric comparison chart."""
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "selection_score",
    ]
    long_df = metrics_df.melt(
        id_vars="model",
        value_vars=metric_columns,
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=long_df,
        x="metric",
        y="score",
        hue="model",
        palette="Set2",
    )
    plt.ylim(0, 1)
    plt.title("Model Comparison Table Visualized", fontsize=15, weight="bold")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.legend(title="Model", loc="lower right")
    _save_current_figure("model_comparison.png")


def plot_roc_curves(evaluation_payload: Dict[str, dict]) -> None:
    """Save ROC curves for all trained models."""
    plt.figure(figsize=(8.5, 6.5))
    for model_name, payload in evaluation_payload.items():
        plt.plot(
            payload["fpr"],
            payload["tpr"],
            linewidth=2,
            label=f"{model_name} (AUC={payload['roc_auc']:.3f})",
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1)
    plt.title("ROC Curve Comparison", fontsize=15, weight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    _save_current_figure("roc_curve.png")


def plot_precision_recall_curves(evaluation_payload: Dict[str, dict]) -> None:
    """Save precision-recall curves for all trained models."""
    plt.figure(figsize=(8.5, 6.5))
    for model_name, payload in evaluation_payload.items():
        plt.plot(
            payload["recall_curve"],
            payload["precision_curve"],
            linewidth=2,
            label=f"{model_name} (F1={payload['f1_score']:.3f})",
        )
    plt.title("Precision-Recall Curve Comparison", fontsize=15, weight="bold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(0, 1.05)
    plt.legend(loc="lower left")
    _save_current_figure("precision_recall_curve.png")


def plot_feature_importance(best_model, model_name: str, top_n: int = 18) -> None:
    """Save feature importance or coefficient magnitude for the selected model."""
    preprocessor = best_model.named_steps["preprocessor"]
    classifier = best_model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out(MODEL_FEATURES)

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        title = f"Feature Importance - {model_name}"
    elif hasattr(classifier, "coef_"):
        importances = abs(classifier.coef_[0])
        title = f"Feature Coefficient Magnitude - {model_name}"
    else:
        return

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=importance_df,
        y="feature",
        x="importance",
        hue="feature",
        palette="crest",
        legend=False,
    )
    plt.title(title, fontsize=15, weight="bold")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    _save_current_figure("feature_importance.png")


def generate_eda_plots(df: pd.DataFrame) -> None:
    """Generate the full exploratory analysis plot suite."""
    _prepare_output_dir()
    plot_class_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df)
    plot_boxplots(df)
    plot_pairplot(df)
