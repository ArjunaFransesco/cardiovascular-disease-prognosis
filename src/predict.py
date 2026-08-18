"""
Clinical Inference & Risk Stratification Engine for Cardiovascular Disease.
Translates patient physiological features into 10-year ASCVD risk categories,
clinical triage recommendations, and actionable risk drivers.
"""

import os
import joblib
import numpy as np
import pandas as pd


class CardiovascularRiskPredictor:
    def __init__(self, model_path="models/best_cardio_model.joblib", transformer_path="models/cardio_transformer.joblib"):
        if not os.path.exists(model_path) or not os.path.exists(transformer_path):
            raise FileNotFoundError("Model or transformer artifact not found. Please run train.py first.")
        
        self.model = joblib.load(model_path)
        self.transformer = joblib.load(transformer_path)

    def stratify_aha_risk(self, prob: float):
        """
        Maps probability to ACC/AHA 10-Year ASCVD Risk Categories.
        """
        risk_pct = round(prob * 100, 1)

        if risk_pct < 10.0:
            category = "Low Risk"
            tier = "AHA Tier 1"
            recommendation = "ROUTINE_MONITORING"
            color = "#10b981"  # Emerald
            action = "Maintain heart-healthy diet, 150 min/wk moderate exercise, routine annual vitals screening."
        elif risk_pct < 25.0:
            category = "Borderline / Moderate Risk"
            tier = "AHA Tier 2"
            recommendation = "LIFESTYLE_OPTIMIZATION"
            color = "#f59e0b"  # Amber
            action = "Target blood pressure < 120/80 mmHg, dietary sodium restriction, lipid panel repeat in 6 months."
        elif risk_pct < 45.0:
            category = "Intermediate Risk"
            tier = "AHA Tier 3"
            recommendation = "STATIN_EVALUATION"
            color = "#f97316"  # Orange
            action = "Consider moderate-intensity statin therapy, Coronary Artery Calcium (CAC) scan evaluation."
        else:
            category = "High / Critical Risk"
            tier = "AHA Tier 4"
            recommendation = "CARDIOLOGY_REFERRAL"
            color = "#ef4444"  # Red
            action = "Urgent cardiologist referral, stress echocardiography, high-intensity statin and anti-hypertensive regimen."

        return risk_pct, category, tier, recommendation, color, action

    def extract_clinical_drivers(self, patient: dict) -> list:
        drivers = []
        trestbps = float(patient.get("trestbps", 120))
        if trestbps >= 140:
            drivers.append(f"Resting Blood Pressure ({trestbps:.0f} mmHg) indicates Stage 2 Hypertension")
        elif trestbps >= 130:
            drivers.append(f"Resting Blood Pressure ({trestbps:.0f} mmHg) indicates Stage 1 Hypertension")

        chol = float(patient.get("chol", 200))
        if chol >= 240:
            drivers.append(f"Serum Cholesterol ({chol:.0f} mg/dL) is in High / Hypercholesterolemia range")
        elif chol >= 200:
            drivers.append(f"Serum Cholesterol ({chol:.0f} mg/dL) is in Borderline High range")

        oldpeak = float(patient.get("oldpeak", 0.0))
        if oldpeak >= 2.0:
            drivers.append(f"Significant ST Depression ({oldpeak:.1f} mm) during peak exercise stress")

        if int(patient.get("ca", 0)) > 0:
            drivers.append(f"Fluoroscopy detected {patient['ca']} major vessel(s) with partial blockage")

        if int(patient.get("exang", 0)) == 1:
            drivers.append("Positive for Exercise-Induced Angina symptomatology")

        if int(patient.get("fbs", 0)) == 1:
            drivers.append("Elevated Fasting Blood Sugar (> 120 mg/dL) indicates prediabetic / diabetic risk")

        return drivers

    def predict_patient(self, patient_dict: dict) -> dict:
        df_input = pd.DataFrame([patient_dict])
        X_trans = self.transformer.transform(df_input)

        prob_val = float(self.model.predict_proba(X_trans)[0, 1])
        risk_pct, category, tier, recommendation, color, action = self.stratify_aha_risk(prob_val)
        drivers = self.extract_clinical_drivers(patient_dict)

        return {
            "disease_probability": round(prob_val, 4),
            "estimated_10yr_risk_percent": risk_pct,
            "risk_category": category,
            "aha_tier": tier,
            "clinical_recommendation": recommendation,
            "actionable_guidance": action,
            "color_code": color,
            "clinical_drivers": drivers
        }


if __name__ == "__main__":
    predictor = CardiovascularRiskPredictor()

    # Low risk sample
    patient_low = {
        "age": 42, "sex": 0, "cp": 1, "trestbps": 115, "chol": 180,
        "fbs": 0, "restecg": 0, "thalach": 168, "exang": 0,
        "oldpeak": 0.2, "slope": 2, "ca": 0, "thal": 2
    }

    # High risk sample
    patient_high = {
        "age": 64, "sex": 1, "cp": 0, "trestbps": 160, "chol": 290,
        "fbs": 1, "restecg": 1, "thalach": 118, "exang": 1,
        "oldpeak": 3.2, "slope": 1, "ca": 2, "thal": 3
    }

    print("\n--- Low Risk Patient Assessment ---")
    print(predictor.predict_patient(patient_low))

    print("\n--- High Risk Patient Assessment ---")
    print(predictor.predict_patient(patient_high))
