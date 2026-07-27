"""Data access and domain feature engineering."""

from typing import Dict

import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TARGET_LABEL_COLUMN,
    TARGET_NAMES,
)


DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)
DATASET_DESCRIPTION_URL = (
    "https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)"
)

RAW_COLUMNS = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_status",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "target",
]

EMPLOYMENT_MAP: Dict[str, str] = {
    "A71": "unemployed",
    "A72": "less than 1 year",
    "A73": "1 to 4 years",
    "A74": "4 to 7 years",
    "A75": "7 or more years",
}

SAVINGS_MAP: Dict[str, str] = {
    "A61": "less than 100 DM",
    "A62": "100 to 500 DM",
    "A63": "500 to 1000 DM",
    "A64": "1000 DM or more",
    "A65": "unknown or no savings account",
}

CREDIT_HISTORY_MAP: Dict[str, str] = {
    "A30": "no credits taken or all paid back duly",
    "A31": "all credits at this bank paid back duly",
    "A32": "existing credits paid back duly till now",
    "A33": "delay in paying off in the past",
    "A34": "critical account or other credits existing",
}

PAYMENT_HISTORY_MAP: Dict[str, str] = {
    "A30": "no prior payment record",
    "A31": "paid previous credits",
    "A32": "paid current credits on time",
    "A33": "past payment delay",
    "A34": "critical account history",
}


def fetch_raw_german_credit(force_download: bool = False) -> pd.DataFrame:
    """Load the German Credit dataset, downloading it if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_PATH.exists() and not force_download:
        return pd.read_csv(RAW_DATA_PATH)

    try:
        raw_df = pd.read_csv(
            DATASET_URL,
            sep=r"\s+",
            header=None,
            names=RAW_COLUMNS,
            engine="python",
        )
    except Exception as exc:
        message = (
            "Unable to download the UCI German Credit dataset. "
            f"Check your internet connection or manually place the file at "
            f"{RAW_DATA_PATH}. Source URL: {DATASET_URL}"
        )
        raise RuntimeError(message) from exc

    raw_df.to_csv(RAW_DATA_PATH, index=False)
    return raw_df


def _map_category(series: pd.Series, mapping: Dict[str, str], field: str) -> pd.Series:
    """Map UCI categorical codes to readable labels."""
    mapped = series.map(mapping)
    if mapped.isna().any():
        missing_values = sorted(series[mapped.isna()].dropna().unique())
        raise ValueError(f"Unexpected values in {field}: {missing_values}")
    return mapped


def create_domain_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw UCI fields into a user-facing credit scoring dataset."""
    df = pd.DataFrame()

    duration = raw_df["duration"].astype(float).clip(lower=1)
    loan_amount = raw_df["credit_amount"].astype(float).clip(lower=1)
    installment_rate = raw_df["installment_rate"].astype(float).clip(lower=1)
    existing_loans = raw_df["existing_credits"].astype(float).clip(lower=1)

    monthly_payment_proxy = loan_amount / duration
    monthly_debt = monthly_payment_proxy * existing_loans * (1 + installment_rate / 10)

    income_multiplier = raw_df["installment_rate"].map(
        {
            1: 6.0,
            2: 4.5,
            3: 3.5,
            4: 2.8,
        }
    )
    employment_multiplier = raw_df["employment_status"].map(
        {
            "A71": 0.75,
            "A72": 0.90,
            "A73": 1.00,
            "A74": 1.15,
            "A75": 1.30,
        }
    )

    annual_income_proxy = monthly_debt * income_multiplier * employment_multiplier * 12

    df["age"] = raw_df["age"].astype(int)
    df["income"] = annual_income_proxy.round(2)
    df["employment_status"] = _map_category(
        raw_df["employment_status"], EMPLOYMENT_MAP, "employment_status"
    )
    df["loan_amount"] = loan_amount.round(2)
    df["debt"] = monthly_debt.round(2)
    df["savings"] = _map_category(raw_df["savings_status"], SAVINGS_MAP, "savings")
    df["credit_history"] = _map_category(
        raw_df["credit_history"], CREDIT_HISTORY_MAP, "credit_history"
    )
    df["payment_history"] = _map_category(
        raw_df["credit_history"], PAYMENT_HISTORY_MAP, "payment_history"
    )
    df["existing_loans"] = existing_loans.astype(int)
    df["loan_duration_months"] = duration.astype(int)

    annual_debt_proxy = df["debt"] * 12
    df["debt_to_income_ratio"] = (
        annual_debt_proxy / df["income"].replace(0, np.nan)
    ).fillna(0).round(4)
    df["loan_to_income_ratio"] = (
        df["loan_amount"] / df["income"].replace(0, np.nan)
    ).fillna(0).round(4)

    df[TARGET_COLUMN] = raw_df["target"].map({1: 1, 2: 0}).astype(int)
    df[TARGET_LABEL_COLUMN] = df[TARGET_COLUMN].map(TARGET_NAMES)
    return df


def load_credit_dataset(force_download: bool = False) -> pd.DataFrame:
    """Return the engineered credit scoring dataset."""
    raw_df = fetch_raw_german_credit(force_download=force_download)
    return create_domain_features(raw_df)


def save_processed_dataset(df: pd.DataFrame) -> None:
    """Persist the processed feature table to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

