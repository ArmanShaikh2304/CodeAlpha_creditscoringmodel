"""Train and evaluate the Credit Scoring Model end to end."""

import json
from datetime import datetime
from typing import Any

import joblib
from sklearn.model_selection import train_test_split

from src.config import (
    CLASSIFICATION_REPORT_PATH,
    METRICS_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    SCREENSHOTS_DIR,
    TARGET_COLUMN,
    TARGET_NAMES,
    TEST_SIZE,
)
from src.data_loader import DATASET_DESCRIPTION_URL, DATASET_URL
from src.data_loader import load_credit_dataset, save_processed_dataset
from src.modeling import select_best_model, train_and_evaluate_models
from src.preprocessing import (
    clean_credit_data,
    display_dataset_overview,
    save_dataset_profile,
    split_features_target,
    validate_no_infinite_values,
)
from src.visualization import (
    generate_eda_plots,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_precision_recall_curves,
    plot_roc_curves,
)


def _json_safe(value: Any) -> Any:
    """Convert numpy and sklearn values into JSON-serializable objects."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    """Execute the complete ML workflow."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading UCI German Credit dataset...")
    df = load_credit_dataset()
    df = clean_credit_data(df)
    validate_no_infinite_values(df)

    save_processed_dataset(df)
    save_dataset_profile(df)

    print("\nDataset overview and summary statistics:")
    display_dataset_overview(df)

    print("\nGenerating EDA plots...")
    generate_eda_plots(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    fitted_models, metrics_df, evaluation_payload = train_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nEvaluation metrics:")
    print(metrics_df.to_string(index=False))

    metrics_df.to_csv(METRICS_PATH, index=False)

    reports = {
        model_name: payload["classification_report"]
        for model_name, payload in evaluation_payload.items()
    }
    CLASSIFICATION_REPORT_PATH.write_text(
        json.dumps(_json_safe(reports), indent=2),
        encoding="utf-8",
    )

    for model_name, payload in evaluation_payload.items():
        plot_confusion_matrix(model_name, payload["confusion_matrix"])

    plot_model_comparison(metrics_df)
    plot_roc_curves(evaluation_payload)
    plot_precision_recall_curves(evaluation_payload)

    best_model_name, best_model = select_best_model(fitted_models, metrics_df)
    plot_feature_importance(best_model, best_model_name)

    model_bundle = {
        "model": best_model,
        "best_model_name": best_model_name,
        "feature_columns": MODEL_FEATURES,
        "target_column": TARGET_COLUMN,
        "target_names": TARGET_NAMES,
        "metrics": metrics_df.to_dict(orient="records"),
        "dataset_source": DATASET_URL,
        "dataset_description": DATASET_DESCRIPTION_URL,
        "trained_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    joblib.dump(model_bundle, MODEL_PATH)

    print(f"\nBest model selected: {best_model_name}")
    print(f"Saved model bundle: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved plots: {SCREENSHOTS_DIR}")


if __name__ == "__main__":
    main()

