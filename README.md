# 🫀 Cardiovascular Disease Risk Prognosis & Clinical AI

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-111?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A Clinical Machine Learning system designed for 10-Year Atherosclerotic Cardiovascular Disease (ASCVD) risk prognosis, biomarker evaluation, ACC/AHA prevention tier stratification, and explainable medical triage recommendations.

---

## 📌 Clinical Overview & Preventive Healthcare Value

Cardiovascular diseases (CVDs) remain the leading global cause of mortality. Early identification of high-risk asymptomatic patients allows clinicians to prescribe targeted statin regimens, blood pressure optimizations, and preventive lifestyle modifications before major adverse cardiac events (MACE) occur.

This pipeline models multi-factor clinical variables (resting blood pressure, serum lipids, exercise ST depression, resting ECG, fluoroscopy vessels, thalassemia defects) to deliver **calibrated 10-year risk probabilities** and **plain-English diagnostic explanations**.

---

## 🏗️ Clinical System Architecture

```
┌───────────────────────────┐
│ Patient Vitals & Records  │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────────┐
│ Feature Engineering (AHA BP Staging, HR Reserve, Ratios)  │
└─────────────┬─────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────────┐
│ Multi-Model Benchmark (Clinical Logistic, RF, GBDT, XGB)   │
└─────────────┬─────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────────┐
│ ACC/AHA 10-Yr Risk Stratification & Clinical Guidance Plan│
└─────────────┬─────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────────┐
│ Production REST API & Interactive MedTech Triage Dashboard │
└───────────────────────────────────────────────────────────┘
```

---

## 📊 Model Benchmark & Diagnostic Metrics

Evaluated on stratified holdout test partitions (600 patients):

| Model Architecture | ROC-AUC | Sensitivity (Recall) | Specificity | F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Clinical Baseline)** 🌟 | **0.7504** | **0.6089** | 0.7411 | **0.5491** | 0.1742 |
| **Random Forest Classifier** | 0.7471 | 0.5698 | 0.7767 | 0.5440 | 0.1765 |
| **Gradient Boosting** | 0.7450 | 0.3352 | **0.8789** | 0.4138 | 0.1770 |
| **XGBoost Classifier** | 0.7434 | 0.5531 | 0.7625 | 0.5238 | 0.1810 |

---

## 🩺 ACC/AHA Clinical Risk Stratification

| 10-Year ASCVD Risk | Clinical Category | AHA Tier | Recommended Clinical Action Plan |
| :---: | :--- | :---: | :--- |
| **< 10%** | **Low Risk** | Tier 1 | Maintain heart-healthy lifestyle, 150 min/wk moderate aerobic exercise, routine annual vitals check. |
| **10% – 24.9%** | **Borderline / Moderate** | Tier 2 | Target blood pressure < 120/80 mmHg, dietary sodium restriction, lipid panel repeat in 6 months. |
| **25% – 44.9%** | **Intermediate Risk** | Tier 3 | Consider moderate-intensity statin therapy, Coronary Artery Calcium (CAC) scan evaluation. |
| **≥ 45%** | **High / Critical Risk** | Tier 4 | Urgent cardiologist referral, stress echocardiography, high-intensity statin & anti-hypertensive regimen. |

---

## 📁 Repository Structure

```
cardiovascular-disease-prognosis/
├── app/
│   ├── static/
│   │   ├── css/style.css       # MedTech dark-mode styling
│   │   └── js/app.js           # Client-side validation & risk calculator
│   ├── templates/
│   │   └── index.html          # Interactive clinical triage dashboard
│   └── main.py                 # Flask server & REST endpoints
├── data/
│   └── raw/
│       └── cardiovascular_risk_dataset.csv # Clinical patient cohort
├── models/
│   ├── best_cardio_model.joblib # Serialized model
│   └── cardio_transformer.joblib # Clinical transformer
├── notebooks/
│   └── cardiovascular_risk_prognosis.ipynb # Full EDA & Model Exploration
├── reports/
│   └── model_evaluation_metrics.json # Automated evaluation report
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Clinical cohort generator & loader
│   ├── features.py             # Feature engineering & AHA BP staging
│   ├── train.py                # Multi-model training & benchmarking
│   └── predict.py              # Clinical inference & triage engine
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusions
└── README.md                   # Documentation
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/ArjunaFransesco/cardiovascular-disease-prognosis.git
cd cardiovascular-disease-prognosis
pip install -r requirements.txt
```

### 2. Train and Benchmark Models
```bash
python src/train.py
```

### 3. Launch Interactive Clinical Triage Dashboard
```bash
python app/main.py
```
Open [http://localhost:5001](http://localhost:5001) in your browser.

---

## 🔌 Clinical REST API Specification

### Endpoint: `POST /api/predict`
Calculates 10-year ASCVD risk for a patient record:

```json
{
  "age": 64,
  "sex": 1,
  "cp": 0,
  "trestbps": 160,
  "chol": 290,
  "fbs": 1,
  "restecg": 1,
  "thalach": 118,
  "exang": 1,
  "oldpeak": 3.2,
  "slope": 1,
  "ca": 2,
  "thal": 3
}
```

#### Response:
```json
{
  "status": "success",
  "data": {
    "disease_probability": 0.8681,
    "estimated_10yr_risk_percent": 86.8,
    "risk_category": "High / Critical Risk",
    "aha_tier": "AHA Tier 4",
    "clinical_recommendation": "CARDIOLOGY_REFERRAL",
    "actionable_guidance": "Urgent cardiologist referral, stress echocardiography, high-intensity statin and anti-hypertensive regimen.",
    "color_code": "#ef4444",
    "clinical_drivers": [
      "Resting Blood Pressure (160 mmHg) indicates Stage 2 Hypertension",
      "Serum Cholesterol (290 mg/dL) is in High / Hypercholesterolemia range",
      "Significant ST Depression (3.2 mm) during peak exercise stress",
      "Fluoroscopy detected 2 major vessel(s) with partial blockage",
      "Positive for Exercise-Induced Angina symptomatology",
      "Elevated Fasting Blood Sugar (> 120 mg/dL) indicates prediabetic / diabetic risk"
    ]
  }
}
```

---

## 👤 Author & Portfolio
- **Author:** [Arjuna Fransesco](https://github.com/ArjunaFransesco)
- **GitHub Repositories:** [https://github.com/ArjunaFransesco?tab=repositories](https://github.com/ArjunaFransesco?tab=repositories)
- **Portfolio Website:** [https://github.com/ArjunaFransesco/arjuna-portfolio](https://github.com/ArjunaFransesco/arjuna-portfolio)


<!-- Last Maintenance Audit: 2026-09-09 -->
