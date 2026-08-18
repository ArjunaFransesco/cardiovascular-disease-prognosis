"""
Data Loader & Dataset Generator for Cardiovascular Disease Risk Prognosis.
Follows clinical cardiology variables based on Cleveland & Framingham Heart Studies.
"""

import os
import numpy as np
import pandas as pd


def generate_cardiovascular_dataset(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a statistically realistic clinical cardiovascular patient cohort
    with realistic physiological correlations and disease markers.
    """
    np.random.seed(random_state)

    # 1. Patient Demographics
    age = np.random.normal(54, 9.5, size=n_samples).astype(int)
    age = np.clip(age, 29, 78)

    # Sex (0 = Female, 1 = Male)
    sex = np.random.binomial(1, 0.68, size=n_samples)

    # Chest Pain Type: 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic
    cp_probs = [0.48, 0.17, 0.28, 0.07]
    cp = np.random.choice([0, 1, 2, 3], size=n_samples, p=cp_probs)

    # Resting Blood Pressure (trestbps in mmHg)
    trestbps = (110 + 0.35 * age + np.random.normal(0, 14, size=n_samples)).astype(int)
    trestbps = np.clip(trestbps, 94, 200)

    # Serum Cholesterol (chol in mg/dl)
    chol = (180 + 0.9 * age + np.random.normal(0, 40, size=n_samples)).astype(int)
    chol = np.clip(chol, 126, 564)

    # Fasting Blood Sugar > 120 mg/dl (fbs: 1 = True, 0 = False)
    fbs_prob = 0.08 + 0.002 * age + 0.0003 * (chol - 200)
    fbs = (np.random.rand(n_samples) < np.clip(fbs_prob, 0.05, 0.40)).astype(int)

    # Resting ECG: 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy
    restecg_probs = [0.49, 0.49, 0.02]
    restecg = np.random.choice([0, 1, 2], size=n_samples, p=restecg_probs)

    # Max Heart Rate (thalach) inversely related to age (220 - age physiological approximation)
    thalach = (205 - 0.85 * age + np.random.normal(0, 18, size=n_samples)).astype(int)
    thalach = np.clip(thalach, 71, 202)

    # Exercise Induced Angina (exang: 1 = Yes, 0 = No)
    exang_prob = 0.20 + 0.003 * age - 0.002 * (thalach - 140) + 0.15 * (cp == 0)
    exang = (np.random.rand(n_samples) < np.clip(exang_prob, 0.05, 0.85)).astype(int)

    # ST Depression induced by exercise (oldpeak)
    oldpeak = np.clip(np.random.exponential(scale=1.0, size=n_samples) * (1 + 0.8 * exang), 0.0, 6.2)
    oldpeak = np.round(oldpeak, 1)

    # Slope of peak exercise ST segment: 0: Upsloping, 1: Flat, 2: Downsloping
    slope_probs = [0.46, 0.47, 0.07]
    slope = np.random.choice([0, 1, 2], size=n_samples, p=slope_probs)

    # Number of major vessels colored by fluoroscopy (ca: 0 to 3)
    ca_prob = [0.58, 0.22, 0.13, 0.07]
    ca = np.random.choice([0, 1, 2, 3], size=n_samples, p=ca_prob)

    # Thalassemia: 1: Normal, 2: Fixed Defect, 3: Reversible Defect
    thal_probs = [0.06, 0.54, 0.40]
    thal = np.random.choice([1, 2, 3], size=n_samples, p=thal_probs)

    # True Diagnostic Risk calculation (Log-odds logistic model)
    z = (
        -4.2
        + 0.045 * (age - 50)
        + 0.65 * sex
        + 0.75 * (cp > 0).astype(int)
        + 0.015 * (trestbps - 120)
        + 0.004 * (chol - 200)
        + 0.40 * fbs
        + 0.012 * (thalach - 130)
        - 0.85 * exang
        + 0.45 * oldpeak
        + 0.55 * (slope == 1).astype(int)
        + 0.70 * ca
        + 0.80 * (thal == 2).astype(int)
    )
    p_disease = 1 / (1 + np.exp(-z))
    target = (np.random.rand(n_samples) < p_disease).astype(int)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "target": target
    })

    return df


def load_data(data_path: str = "data/raw/cardiovascular_risk_dataset.csv", n_samples: int = 3000) -> pd.DataFrame:
    """
    Loads dataset from local CSV or generates fresh benchmark dataset if not found.
    """
    if os.path.exists(data_path):
        print(f"[+] Loading cardiovascular dataset from {data_path}")
        return pd.read_csv(data_path)

    print(f"[!] Dataset not found at {data_path}. Generating {n_samples} clinical records...")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df = generate_cardiovascular_dataset(n_samples=n_samples)
    df.to_csv(data_path, index=False)
    print(f"[+] Clinical dataset generated and saved to {data_path}")
    return df


if __name__ == "__main__":
    df = load_data()
    print("Dataset Shape:", df.shape)
    print("Prevalence of Heart Disease Target:\n", df["target"].value_counts(normalize=True))
    print("\nSample Patient Records:")
    print(df.head())
