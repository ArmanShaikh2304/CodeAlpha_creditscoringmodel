"""Command-line inference for the saved Credit Scoring Model."""

import argparse
from pathlib import Path

import joblib

from src.config import (
    CREDIT_HISTORY_OPTIONS,
    EMPLOYMENT_OPTIONS,
    MODEL_PATH,
    SAVINGS_OPTIONS,
    TARGET_NAMES,
)
from src.preprocessing import build_prediction_input


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a single applicant prediction."""
    parser = argparse.ArgumentParser(
        description="Predict whether a credit applicant is creditworthy."
    )
    parser.add_argument("--age", type=int, default=35)
    parser.add_argument("--income", type=float, default=52000.0)
    parser.add_argument("--employment-status", default="1 to 4 years")
    parser.add_argument("--loan-amount", type=float, default=7500.0)
    parser.add_argument("--debt", type=float, default=750.0)
    parser.add_argument("--savings", default="100 to 500 DM")
    parser.add_argument(
        "--credit-history",
        default="existing credits paid back duly till now",
    )
    parser.add_argument("--existing-loans", type=int, default=1)
    parser.add_argument("--loan-duration-months", type=int, default=24)
    return parser.parse_args()


def load_model_bundle(model_path: Path = MODEL_PATH) -> dict:
    """Load the trained model artifact."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python train.py` first."
        )
    return joblib.load(model_path)


def predict_creditworthiness(args: argparse.Namespace) -> None:
    """Run prediction and print a clear result."""
    if args.employment_status not in EMPLOYMENT_OPTIONS:
        raise ValueError(f"Invalid employment status: {args.employment_status}")
    if args.savings not in SAVINGS_OPTIONS:
        raise ValueError(f"Invalid savings value: {args.savings}")
    if args.credit_history not in CREDIT_HISTORY_OPTIONS:
        raise ValueError(f"Invalid credit history: {args.credit_history}")

    bundle = load_model_bundle()
    model = bundle["model"]
    applicant = build_prediction_input(
        age=args.age,
        income=args.income,
        employment_status=args.employment_status,
        loan_amount=args.loan_amount,
        debt=args.debt,
        savings=args.savings,
        credit_history=args.credit_history,
        existing_loans=args.existing_loans,
        loan_duration_months=args.loan_duration_months,
    )

    predicted_label = int(model.predict(applicant)[0])
    probabilities = model.predict_proba(applicant)[0]
    probability_by_label = {
        TARGET_NAMES[index]: float(probability)
        for index, probability in enumerate(probabilities)
    }
    prediction = TARGET_NAMES[predicted_label]
    confidence = probability_by_label[prediction]

    print("\nCredit Scoring Prediction")
    print("=" * 28)
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Good Credit Probability: {probability_by_label['Good Credit']:.2%}")
    print(f"Bad Credit Probability: {probability_by_label['Bad Credit']:.2%}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    predict_creditworthiness(args)


if __name__ == "__main__":
    main()

