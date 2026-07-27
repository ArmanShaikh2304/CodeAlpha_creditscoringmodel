"""Preprocessing utilities for model training and inference."""

from io import StringIO
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PROFILE_PATH,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TARGET_LABEL_COLUMN,
)


def clean_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and normalize feature dtypes."""
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for column in NUMERIC_FEATURES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        cleaned[column] = cleaned[column].astype("category")

    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    return cleaned


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the binary target."""
    return df[MODEL_FEATURES].copy(), df[TARGET_COLUMN].copy()


def create_preprocessor() -> ColumnTransformer:
    """Build a preprocessing transformer for numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_prediction_input(
    age: int,
    income: float,
    employment_status: str,
    loan_amount: float,
    debt: float,
    savings: str,
    credit_history: str,
    existing_loans: int = 1,
    loan_duration_months: int = 24,
) -> pd.DataFrame:
    """Create a one-row feature frame that matches the training schema."""
    safe_income = max(float(income), 1.0)
    safe_debt = max(float(debt), 0.0)
    safe_loan_amount = max(float(loan_amount), 0.0)
    safe_duration = max(int(loan_duration_months), 1)
    safe_existing_loans = max(int(existing_loans), 0)

    row = {
        "age": int(age),
        "income": safe_income,
        "employment_status": employment_status,
        "loan_amount": safe_loan_amount,
        "debt": safe_debt,
        "savings": savings,
        "credit_history": credit_history,
        "existing_loans": safe_existing_loans,
        "loan_duration_months": safe_duration,
        "debt_to_income_ratio": round((safe_debt * 12) / safe_income, 4),
        "loan_to_income_ratio": round(safe_loan_amount / safe_income, 4),
    }

    return pd.DataFrame([row], columns=MODEL_FEATURES)


def dataset_profile(df: pd.DataFrame) -> str:
    """Create a text profile with shape, schema, missing values, and statistics."""
    buffer = StringIO()
    buffer.write("Credit Scoring Dataset Profile\n")
    buffer.write("=" * 36 + "\n\n")
    buffer.write(f"Rows: {df.shape[0]}\n")
    buffer.write(f"Columns: {df.shape[1]}\n")
    buffer.write(f"Duplicate rows: {df.duplicated().sum()}\n\n")

    buffer.write("Dataset Info\n")
    buffer.write("-" * 12 + "\n")
    info_buffer = StringIO()
    df.info(buf=info_buffer)
    buffer.write(info_buffer.getvalue())
    buffer.write("\n")

    buffer.write("Missing Values\n")
    buffer.write("-" * 14 + "\n")
    missing = df.isna().sum()
    buffer.write(missing.to_string())
    buffer.write("\n\n")

    buffer.write("Summary Statistics\n")
    buffer.write("-" * 18 + "\n")
    buffer.write(df.describe(include="all").transpose().to_string())
    buffer.write("\n\n")

    buffer.write("Class Distribution\n")
    buffer.write("-" * 18 + "\n")
    class_distribution = df[TARGET_LABEL_COLUMN].value_counts()
    buffer.write(class_distribution.to_string())
    buffer.write("\n")
    return buffer.getvalue()


def display_dataset_overview(df: pd.DataFrame) -> None:
    """Print dataset information and summary statistics to the terminal."""
    print(dataset_profile(df))


def save_dataset_profile(df: pd.DataFrame) -> None:
    """Save the dataset profile for submission evidence."""
    DATA_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PROFILE_PATH.write_text(dataset_profile(df), encoding="utf-8")


def validate_no_infinite_values(df: pd.DataFrame) -> None:
    """Fail early if engineered numeric features contain invalid infinite values."""
    numeric_frame = df.select_dtypes(include=[np.number])
    if np.isinf(numeric_frame.to_numpy()).any():
        raise ValueError("Infinite values found in numeric features.")

