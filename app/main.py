"""
Flask Web Application & Clinical REST API for Cardiovascular Disease Risk Prognosis.
Serves real-time 10-year ASCVD risk stratification and clinical triage explanations.
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, jsonify, render_template, request
from src.predict import CardiovascularRiskPredictor

app = Flask(__name__)

# Initialize predictor
try:
    predictor = CardiovascularRiskPredictor(
        model_path=os.path.join(os.path.dirname(__file__), "../models/best_cardio_model.joblib"),
        transformer_path=os.path.join(os.path.dirname(__file__), "../models/cardio_transformer.joblib")
    )
except Exception as e:
    print(f"[!] Warning: Initializing predictor from default paths ({e})")
    predictor = CardiovascularRiskPredictor()

# Load metrics report
metrics_path = os.path.join(os.path.dirname(__file__), "../reports/model_evaluation_metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        MODEL_STATS = json.load(f)
else:
    MODEL_STATS = {}


@app.route("/")
def home():
    return render_template("index.html", stats=MODEL_STATS)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No patient record provided"}), 400

        result = predictor.predict_patient(data)
        return jsonify({
            "status": "success",
            "data": result
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(MODEL_STATS)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "Cardiovascular Prognosis Engine v1.0"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"[*] Starting Cardiovascular Prognosis Dashboard on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
