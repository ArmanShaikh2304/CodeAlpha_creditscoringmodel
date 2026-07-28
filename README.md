<div align="center">

# 💳 Credit Scoring Model

### An End-to-End Machine Learning Pipeline for Predicting Applicant Creditworthiness

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)]()

*A complete Machine Learning project that predicts whether a loan applicant is likely to be **creditworthy** using multiple classification algorithms, advanced preprocessing, feature engineering, model evaluation, and an interactive Streamlit dashboard.*

</div>

---

# 🌟 Project Overview

This project demonstrates a complete **Machine Learning workflow** from raw data to deployment.

Using the **UCI German Credit Dataset**, the application predicts whether an applicant has **Good Credit** or **Bad Credit** based on financial and demographic information.

The project includes everything required in a production-style ML pipeline:

- Data Collection
- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- Automatic Best Model Selection
- Model Serialization
- Command-Line Prediction
- Interactive Streamlit Web Application

---

# ✨ Features

### 📊 Data Processing

- Public UCI German Credit Dataset
- Missing Value Handling
- Duplicate Removal
- Feature Encoding
- Feature Scaling
- Automated Dataset Profiling

---

### 🧠 Machine Learning

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Cross Validation
- GridSearchCV Hyperparameter Optimization
- Automatic Best Model Selection

---

### 📈 Model Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report
- ROC Curve
- Precision-Recall Curve

---

### 🎯 Prediction

- Command Line Prediction
- Interactive Streamlit Dashboard
- Credit Probability Score
- Prediction Confidence
- Risk Classification

---

### 📉 Data Visualization

- Class Distribution
- Correlation Heatmap
- Feature Distribution
- Boxplots
- Pairplots
- Feature Importance
- Model Comparison Charts
- ROC Curve
- Precision-Recall Curve

---

# 🧠 Machine Learning Pipeline

```text
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Train Multiple Models
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
Model Serialization
      │
      ▼
Prediction API
      │
      ▼
Streamlit Web Application
```

---

# ⚙️ Technologies Used

| Category | Technologies |
|-----------|--------------|
| Language | Python |
| Machine Learning | Scikit-Learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Saving | Joblib |
| Web Application | Streamlit |
| IDE | VS Code |

---

# 📂 Dataset

**Dataset:** German Credit Dataset

The model is trained on the **UCI Statlog German Credit Dataset**, a widely used benchmark dataset for binary credit classification.

### Target Labels

| Original | Meaning |
|----------|---------|
| **1** | Good Credit |
| **2** | Bad Credit |

To improve prediction quality, additional financial indicators were engineered, including:

- Income Proxy
- Debt
- Debt-to-Income Ratio
- Loan-to-Income Ratio

---

# 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------|---------:|----------:|--------:|---------:|--------:|
| 🌲 Random Forest | **68.50%** | **73.33%** | **86.43%** | **79.34%** | **69.57%** |
| 📈 Logistic Regression | 66.00% | **83.96%** | 63.57% | 72.36% | **76.37%** |
| 🌳 Decision Tree | **70.00%** | 76.32% | 82.86% | **79.45%** | 63.37% |

---

# 🏆 Best Model

The project automatically selects the best-performing model using:

```python
Selection Score = (F1 Score + ROC-AUC) / 2
```

🏅 **Selected Model:** **Random Forest Classifier**

---

# 📊 Visual Outputs

The project generates multiple visualizations for better model interpretation.

- 📈 Model Comparison
- 📊 Feature Importance
- 📉 Correlation Heatmap
- 📦 Feature Distribution
- 📊 Boxplots
- 📈 Pairplots
- 📉 ROC Curve
- 📊 Precision-Recall Curve
- 🎯 Confusion Matrix
- 📋 Dataset Profile

---

# 🚀 Key Highlights

✅ End-to-End Machine Learning Project

✅ Complete Data Preprocessing Pipeline

✅ Feature Engineering

✅ Hyperparameter Tuning

✅ Multiple ML Algorithms

✅ Automatic Best Model Selection

✅ Interactive Streamlit Dashboard

✅ Production-Ready Model Serialization

✅ Command Line Prediction Support

✅ Professional Data Visualization

---

# 💡 Future Improvements

- 🤖 Deep Learning-based Credit Risk Prediction
- ⚖️ Fairness & Bias Detection
- 📊 MLflow Experiment Tracking
- 🌐 REST API using FastAPI
- ☁️ Cloud Deployment
- 📈 Real-Time Prediction Service
- 🔄 Continuous Model Monitoring
- 🧪 Automated Model Validation

---

# 👨‍💻 Author

## **Arman Shaikh**

**Computer Science Engineering Student**

Passionate about

- 🤖 Machine Learning
- 🧠 Artificial Intelligence
- 🌐 Full Stack Development
- 📊 Data Science
- 🚀 Open Source

---

<div align="center">

## ⭐ If you found this project helpful, don't forget to Star the repository!

### Made with ❤️ using Python, Scikit-Learn, Streamlit & Machine Learning

</div>
