"""Streamlit application for interactive credit scoring."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    CREDIT_HISTORY_OPTIONS,
    EMPLOYMENT_OPTIONS,
    MODEL_PATH,
    SAVINGS_OPTIONS,
    SCREENSHOTS_DIR,
    TARGET_NAMES,
)
from src.preprocessing import build_prediction_input


st.set_page_config(
    page_title="Credit Scoring Model",
    layout="wide",
)


@st.cache_resource
def load_model_bundle() -> dict:
    """Load and cache the trained model bundle."""
    if not MODEL_PATH.exists():
        return {}
    return joblib.load(MODEL_PATH)


def _metric_card(label: str, value: str) -> None:
    """Render a compact metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <span>{label}</span>
            <strong>{value}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _load_metrics(bundle: dict) -> pd.DataFrame:
    """Return the saved model comparison table if present."""
    metrics = bundle.get("metrics", [])
    return pd.DataFrame(metrics)


def main() -> None:
    """Render the Streamlit application."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            max-width: 1180px;
        }
        .app-title {
            font-size: 2.2rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }
        .subtle {
            color: #56616f;
            font-size: 1rem;
            margin-bottom: 1.2rem;
        }
        .metric-card {
            border: 1px solid #d9e1ea;
            border-radius: 8px;
            padding: 1rem;
            background: #ffffff;
        }
        .metric-card span {
            display: block;
            color: #657083;
            font-size: 0.86rem;
        }
        .metric-card strong {
            display: block;
            color: #111827;
            font-size: 1.45rem;
            margin-top: 0.25rem;
        }
        .prediction-good {
            border-left: 6px solid #178f5b;
            background: #eefaf4;
            padding: 1rem 1.1rem;
            border-radius: 8px;
        }
        .prediction-bad {
            border-left: 6px solid #c43d3d;
            background: #fff1f1;
            padding: 1rem 1.1rem;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="app-title">Credit Scoring Model</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtle">Assess applicant creditworthiness with a trained '
        "machine learning pipeline.</div>",
        unsafe_allow_html=True,
    )

    bundle = load_model_bundle()
    if not bundle:
        st.warning("Model artifact not found. Run `python train.py` before launching the app.")
        st.stop()

    model = bundle["model"]
    best_model_name = bundle.get("best_model_name", "Trained model")

    input_col, output_col = st.columns([0.95, 1.05], gap="large")

    with input_col:
        st.subheader("Applicant Profile")
        age = st.slider("Age", min_value=18, max_value=80, value=35)
        income = st.number_input(
            "Annual Income",
            min_value=1000.0,
            max_value=500000.0,
            value=52000.0,
            step=1000.0,
        )
        employment_status = st.selectbox(
            "Employment Status",
            EMPLOYMENT_OPTIONS,
            index=2,
        )
        loan_amount = st.number_input(
            "Loan Amount",
            min_value=100.0,
            max_value=250000.0,
            value=7500.0,
            step=250.0,
        )
        debt = st.number_input(
            "Monthly Debt",
            min_value=0.0,
            max_value=50000.0,
            value=750.0,
            step=50.0,
        )
        savings = st.selectbox("Savings", SAVINGS_OPTIONS, index=1)
        credit_history = st.selectbox(
            "Credit History",
            CREDIT_HISTORY_OPTIONS,
            index=2,
        )

        with st.expander("Advanced loan details"):
            existing_loans = st.number_input(
                "Existing Loans",
                min_value=0,
                max_value=10,
                value=1,
                step=1,
            )
            loan_duration_months = st.slider(
                "Loan Duration in Months",
                min_value=4,
                max_value=72,
                value=24,
            )

        submit = st.button(
            "Predict Creditworthiness",
            type="primary",
            width="stretch",
        )

    applicant = build_prediction_input(
        age=age,
        income=income,
        employment_status=employment_status,
        loan_amount=loan_amount,
        debt=debt,
        savings=savings,
        credit_history=credit_history,
        existing_loans=existing_loans,
        loan_duration_months=loan_duration_months,
    )

    with output_col:
        st.subheader("Prediction")
        if submit:
            predicted_label = int(model.predict(applicant)[0])
            probabilities = model.predict_proba(applicant)[0]
            probability_by_label = {
                TARGET_NAMES[index]: float(probability)
                for index, probability in enumerate(probabilities)
            }
            prediction = TARGET_NAMES[predicted_label]
            confidence = probability_by_label[prediction]
            prediction_class = (
                "prediction-good" if prediction == "Good Credit" else "prediction-bad"
            )

            st.markdown(
                f"""
                <div class="{prediction_class}">
                    <h3 style="margin-top: 0;">{prediction}</h3>
                    <p style="margin-bottom: 0;">
                        Confidence: <strong>{confidence:.2%}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric_cols = st.columns(3)
            with metric_cols[0]:
                _metric_card("Good Credit Probability", f"{probability_by_label['Good Credit']:.2%}")
            with metric_cols[1]:
                _metric_card("Bad Credit Probability", f"{probability_by_label['Bad Credit']:.2%}")
            with metric_cols[2]:
                _metric_card("Selected Model", best_model_name)

            st.progress(float(confidence))
        else:
            st.info("Enter applicant details and run the prediction.")

        st.divider()
        st.subheader("Model Comparison")
        metrics_df = _load_metrics(bundle)
        if not metrics_df.empty:
            display_columns = [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "selection_score",
            ]
            st.dataframe(
                metrics_df[display_columns],
                hide_index=True,
                width="stretch",
            )

        feature_importance_path = SCREENSHOTS_DIR / "feature_importance.png"
        if Path(feature_importance_path).exists():
            st.image(str(feature_importance_path), caption="Top Model Drivers")


if __name__ == "__main__":
    main()
