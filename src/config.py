"""Shared configuration for the Credit Scoring Model project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

RAW_DATA_PATH = DATA_DIR / "german_credit_raw.csv"
PROCESSED_DATA_PATH = DATA_DIR / "credit_scoring_processed.csv"
DATA_PROFILE_PATH = DATA_DIR / "dataset_profile.txt"

MODEL_PATH = MODELS_DIR / "credit_model.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.csv"
CLASSIFICATION_REPORT_PATH = MODELS_DIR / "classification_reports.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_JOBS = 1

TARGET_COLUMN = "creditworthy"
TARGET_LABEL_COLUMN = "credit_status"
TARGET_NAMES = {
    0: "Bad Credit",
    1: "Good Credit",
}

NUMERIC_FEATURES = [
    "age",
    "income",
    "loan_amount",
    "debt",
    "existing_loans",
    "loan_duration_months",
    "debt_to_income_ratio",
    "loan_to_income_ratio",
]

CATEGORICAL_FEATURES = [
    "employment_status",
    "savings",
    "credit_history",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EMPLOYMENT_OPTIONS = [
    "unemployed",
    "less than 1 year",
    "1 to 4 years",
    "4 to 7 years",
    "7 or more years",
]

SAVINGS_OPTIONS = [
    "less than 100 DM",
    "100 to 500 DM",
    "500 to 1000 DM",
    "1000 DM or more",
    "unknown or no savings account",
]

CREDIT_HISTORY_OPTIONS = [
    "no credits taken or all paid back duly",
    "all credits at this bank paid back duly",
    "existing credits paid back duly till now",
    "delay in paying off in the past",
    "critical account or other credits existing",
]
