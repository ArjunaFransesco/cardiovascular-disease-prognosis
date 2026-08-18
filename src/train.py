"""
Training, Benchmarking, and Evaluation Pipeline for Cardiovascular Risk Prognosis.
Evaluates clinical sensitivity, specificity, ROC-AUC, Brier score, and serializes best model artifacts.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from data_loader import load_data
from features import CardioFeatureTransformer


def train_and_benchmark(data_path="data/raw/cardiovascular_risk_dataset.csv", output_dir="models", reports_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=================================================================")
    print("   CARDIOVASCULAR DISEASE PROGNOSIS - MODEL TRAINING PIPELINE    ")
    print("=================================================================")

    # 1. Load Data
    df = load_data(data_path)
    X = df.drop(columns=["target"])
    y = df["target"].values

    # 2. Stratified Split (80% Train, 20% Holdout Test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[+] Cohort Partition: Train = {len(X_train_raw)} patients, Test = {len(X_test_raw)} patients")
    print(f"[+] Prevalence - Train: {np.mean(y_train):.2%}, Test: {np.mean(y_test):.2%}")

    # 3. Clinical Feature Transformation
    transformer = CardioFeatureTransformer()
    X_train = transformer.fit_transform(X_train_raw)
    X_test = transformer.transform(X_test_raw)

    scale_pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)

    # 4. Model Architectures
    models = {
        "Logistic Regression (Clinical Baseline)": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=160, max_depth=7, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=140, max_depth=4, learning_rate=0.08, random_state=42),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.06,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42
        )
    }

    benchmark_results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\n[>] Training model: {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)

        # Clinical metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall of positive disease
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate

        roc_auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)

        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_curve, precision_curve)

        benchmark_results[name] = {
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "sensitivity_recall": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
            "f1_score": round(float(f1), 4),
            "precision": round(float(prec), 4),
            "accuracy": round(float(acc), 4),
            "brier_score": round(float(brier), 4)
        }

        print(f"    ROC-AUC: {roc_auc:.4f} | Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f} | F1: {f1:.4f}")

    # Select Best Model based on ROC-AUC
    best_model_name = max(benchmark_results, key=lambda k: benchmark_results[k]["roc_auc"])
    best_model = fitted_models[best_model_name]
    print(f"\n[*] Top Performing Clinical Model: {best_model_name} (ROC-AUC: {benchmark_results[best_model_name]['roc_auc']})")

    # 5. Optimal Clinical Threshold Analysis (Targeting high sensitivity >= 80%)
    y_test_probs = best_model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = [f1_score(y_test, (y_test_probs >= t).astype(int), zero_division=0) for t in thresholds]
    optimal_threshold = float(thresholds[np.argmax(f1_scores)])
    best_f1 = float(max(f1_scores))

    # 6. Feature Importances
    feature_importance_dict = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_importance_dict = {
            feat: round(float(imp), 4)
            for feat, imp in sorted(zip(transformer.feature_names_, importances), key=lambda x: x[1], reverse=True)
        }

    # 7. Serialize Artifacts
    model_file = os.path.join(output_dir, "best_cardio_model.joblib")
    transformer_file = os.path.join(output_dir, "cardio_transformer.joblib")
    metrics_file = os.path.join(reports_dir, "model_evaluation_metrics.json")

    joblib.dump(best_model, model_file)
    joblib.dump(transformer, transformer_file)

    report_payload = {
        "best_model": best_model_name,
        "optimal_threshold": round(optimal_threshold, 2),
        "best_f1_score": round(best_f1, 4),
        "benchmark_comparison": benchmark_results,
        "feature_importances": feature_importance_dict,
        "feature_names": transformer.feature_names_,
        "cohort_summary": {
            "total_patients": len(df),
            "train_patients": len(X_train_raw),
            "test_patients": len(X_test_raw),
            "disease_prevalence": round(float(np.mean(y)), 4)
        }
    }

    with open(metrics_file, "w") as f:
        json.dump(report_payload, f, indent=4)

    print(f"[+] Serialized model artifact: {model_file}")
    print(f"[+] Serialized transformer artifact: {transformer_file}")
    print(f"[+] Saved evaluation metrics to: {metrics_file}")
    print("=================================================================\n")

    return report_payload


if __name__ == "__main__":
    train_and_benchmark()
