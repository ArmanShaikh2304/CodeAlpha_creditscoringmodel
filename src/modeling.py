"""Model training, tuning, evaluation, and selection."""

from typing import Dict, Tuple

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import CV_FOLDS, N_JOBS, RANDOM_STATE
from src.preprocessing import create_preprocessor


def build_model_searches() -> Dict[str, GridSearchCV]:
    """Create GridSearchCV objects for each candidate classifier."""
    base_preprocessor = create_preprocessor()
    model_specs = {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "params": {
                "classifier__C": [0.1, 1.0, 3.0, 10.0],
                "classifier__penalty": ["l1", "l2"],
            },
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            "params": {
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [3, 5, 7, None],
                "classifier__min_samples_split": [2, 10, 20],
                "classifier__min_samples_leaf": [1, 5, 10],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=N_JOBS,
            ),
            "params": {
                "classifier__n_estimators": [150, 250],
                "classifier__max_depth": [5, 10, None],
                "classifier__min_samples_leaf": [1, 3, 5],
                "classifier__max_features": ["sqrt", "log2"],
            },
        },
    }

    searches = {}
    for model_name, spec in model_specs.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(base_preprocessor)),
                ("classifier", spec["estimator"]),
            ]
        )
        searches[model_name] = GridSearchCV(
            estimator=pipeline,
            param_grid=spec["params"],
            scoring="f1",
            cv=CV_FOLDS,
            n_jobs=N_JOBS,
            refit=True,
            return_train_score=False,
        )
    return searches


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a fitted model on holdout data."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Bad Credit", "Good Credit"],
            zero_division=0,
            output_dict=True,
        ),
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "predicted_probabilities": y_proba,
    }


def train_and_evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, dict]]:
    """Fit all model searches and return fitted models, metrics, and curves."""
    fitted_models = {}
    metrics_rows = []
    evaluation_payload = {}

    for model_name, search in build_model_searches().items():
        print(f"\nTraining {model_name}...")
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        evaluation = evaluate_model(best_model, X_test, y_test)
        selection_score = (evaluation["f1_score"] + evaluation["roc_auc"]) / 2

        fitted_models[model_name] = best_model
        evaluation_payload[model_name] = evaluation

        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": round(evaluation["accuracy"], 4),
                "precision": round(evaluation["precision"], 4),
                "recall": round(evaluation["recall"], 4),
                "f1_score": round(evaluation["f1_score"], 4),
                "roc_auc": round(evaluation["roc_auc"], 4),
                "selection_score": round(selection_score, 4),
                "best_cv_f1": round(search.best_score_, 4),
                "best_params": search.best_params_,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by=["selection_score", "f1_score", "roc_auc"],
        ascending=False,
    )
    return fitted_models, metrics_df, evaluation_payload


def select_best_model(
    fitted_models: Dict[str, object],
    metrics_df: pd.DataFrame,
) -> Tuple[str, object]:
    """Select the best model using a combined F1 and ROC-AUC score."""
    best_model_name = metrics_df.iloc[0]["model"]
    return best_model_name, fitted_models[best_model_name]
