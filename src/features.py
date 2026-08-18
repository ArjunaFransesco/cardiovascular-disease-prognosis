"""
Clinical Feature Engineering and Biomarker Preprocessor.
Engineers AHA/ACC blood pressure stages, heart rate reserves, cholesterol ratios, and sklearn transformers.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class CardioFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compliant feature transformer for cardiovascular clinical risk assessment.
    """
    def __init__(self):
        self.scaler_ = StandardScaler()
        self.feature_names_ = []

    def fit(self, X, y=None):
        df = self._transform_df(X)
        self.feature_names_ = list(df.columns)
        self.scaler_.fit(df)
        return self

    def _transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. AHA Blood Pressure Categorization
        # Stage 0: Normal (<120), 1: Elevated (120-129), 2: Stage 1 HTN (130-139), 3: Stage 2 HTN (>=140)
        df["bp_stage"] = np.where(
            df["trestbps"] < 120, 0,
            np.where(df["trestbps"] < 130, 1,
            np.where(df["trestbps"] < 140, 2, 3))
        )

        # 2. Physiological Derived Ratios
        estimated_max_hr = 220 - df["age"]
        df["hr_reserve_ratio"] = df["thalach"] / np.maximum(estimated_max_hr, 100)
        df["chol_risk_ratio"] = df["chol"] / 200.0
        df["st_depression_high"] = (df["oldpeak"] >= 2.0).astype(int)

        # 3. Categorical Encodings
        # Chest pain type dummies (cp: 0, 1, 2, 3)
        for cp_val in [0, 1, 2, 3]:
            df[f"cp_type_{cp_val}"] = (df["cp"] == cp_val).astype(int)

        # Resting ECG dummies (restecg: 0, 1, 2)
        for ecg_val in [0, 1, 2]:
            df[f"restecg_{ecg_val}"] = (df["restecg"] == ecg_val).astype(int)

        # Slope dummies (slope: 0, 1, 2)
        for sl_val in [0, 1, 2]:
            df[f"slope_{sl_val}"] = (df["slope"] == sl_val).astype(int)

        # Thalassemia dummies (thal: 1, 2, 3)
        for th_val in [1, 2, 3]:
            df[f"thal_{th_val}"] = (df["thal"] == th_val).astype(int)

        # Drop original multi-category columns
        drop_cols = ["cp", "restecg", "slope", "thal"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        return df

    def transform(self, X):
        df = self._transform_df(X)
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names_]
        return self.scaler_.transform(df)

    def transform_to_df(self, X):
        df = self._transform_df(X)
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_names_]


if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    X = df.drop(columns=["target"])
    fe = CardioFeatureTransformer()
    X_trans = fe.fit_transform(X)
    print(f"[+] Transformed feature matrix shape: {X_trans.shape}")
    print(f"[+] Feature columns ({len(fe.feature_names_)}): {fe.feature_names_}")
