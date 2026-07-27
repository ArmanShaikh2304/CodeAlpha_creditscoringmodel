# Credit Scoring Model

## Project Overview

Credit Scoring Model is an end-to-end machine learning project that predicts
whether an applicant is likely to be creditworthy. The workflow covers data
download, preprocessing, exploratory data analysis, model training,
hyperparameter tuning, evaluation, model selection, command-line prediction,
and a Streamlit web application.

The project uses the public UCI Statlog German Credit dataset and engineers a
compact applicant profile suitable for a user-facing credit scoring interface.

## Features

- Public dataset ingestion from the UCI Machine Learning Repository
- Missing-value handling, duplicate removal, encoding, and numeric scaling
- Domain feature engineering for income, debt, debt-to-income ratio, and
  loan-to-income ratio
- EDA charts saved to `screenshots/`
- Logistic Regression, Decision Tree, and Random Forest training
- Cross-validation and GridSearchCV hyperparameter tuning
- Accuracy, precision, recall, F1 score, ROC-AUC, confusion matrix, and
  classification report for every model
- Automatic best-model selection by F1 score and ROC-AUC
- Saved Joblib model bundle at `models/credit_model.pkl`
- Command-line prediction script
- Streamlit application with probability and confidence output
- Feature importance, ROC curve, precision-recall curve, and model comparison
  charts

## Technologies Used

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- Joblib
- Streamlit

## Dataset Source

- Dataset: UCI Statlog German Credit Data
- Source file: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
- Repository page: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

The original target labels are mapped as:

- `1` -> `Good Credit`
- `2` -> `Bad Credit`

The original dataset does not include a direct income column, so this project
uses transparent proxy income and debt features engineered from credit amount,
loan duration, installment rate, employment status, and existing credits.

## Installation

```bash
cd CodeAlpha_CreditScoring
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

Create the processed dataset and dataset profile:

```bash
python preprocess.py
```

Train models, generate plots, compare model performance, and save the best
model:

```bash
python train.py
```

Run a sample command-line prediction:

```bash
python predict.py
```

Run a custom command-line prediction:

```bash
python predict.py \
  --age 42 \
  --income 68000 \
  --employment-status "4 to 7 years" \
  --loan-amount 12000 \
  --debt 900 \
  --savings "500 to 1000 DM" \
  --credit-history "existing credits paid back duly till now" \
  --existing-loans 2 \
  --loan-duration-months 36
```

Launch the Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

```text
CodeAlpha_CreditScoring/
├── app/
├── data/
├── models/
├── notebooks/
├── screenshots/
├── src/
├── app.py
├── predict.py
├── preprocess.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Model Performance

Latest reproducible holdout results from `python train.py`:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Selection Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.6850 | 0.7333 | 0.8643 | 0.7934 | 0.6957 | 0.7446 |
| Logistic Regression | 0.6600 | 0.8396 | 0.6357 | 0.7236 | 0.7637 | 0.7436 |
| Decision Tree | 0.7000 | 0.7632 | 0.8286 | 0.7945 | 0.6337 | 0.7141 |

The best model is selected automatically using:

```text
selection_score = (F1 Score + ROC-AUC) / 2
```

The saved production artifact is the Random Forest model. The training script
also saves the comparison table to `models/model_metrics.csv` and classification
reports to `models/classification_reports.json`.

## Generated Outputs

After training, the project creates:

- `data/german_credit_raw.csv`
- `data/credit_scoring_processed.csv`
- `data/dataset_profile.txt`
- `models/credit_model.pkl`
- `models/model_metrics.csv`
- `models/classification_reports.json`
- `screenshots/class_distribution.png`
- `screenshots/correlation_heatmap.png`
- `screenshots/feature_distributions.png`
- `screenshots/boxplots_by_credit_status.png`
- `screenshots/pairplot_key_features.png`
- `screenshots/model_comparison.png`
- `screenshots/feature_importance.png`
- `screenshots/roc_curve.png`
- `screenshots/precision_recall_curve.png`
- `screenshots/confusion_matrix_*.png`

## Future Improvements

- Add cost-sensitive threshold tuning for lending-specific risk appetite
- Track experiments with MLflow or Weights & Biases
- Add fairness and bias diagnostics across demographic segments
- Package the pipeline behind a FastAPI service
- Add CI checks for linting, tests, and model artifact validation
- Monitor model drift once production scoring data becomes available
