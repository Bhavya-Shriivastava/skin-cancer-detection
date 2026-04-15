"""
DermAI — Flask API Server
Wraps hybrid_predict() and exposes a single POST /predict endpoint.

Install dependencies:
    pip install flask flask-cors tensorflow pillow numpy

Run:
    python app.py
    → Serves on http://localhost:5000
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import io
import base64
import tempfile
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

# ── Config ───────────────────────────────────────────────────────────
IMG_SIZE    = 224
MODEL_PATH = "cnn_scratch_binary.keras"
HOST        = "0.0.0.0"
PORT        = 5000
DEBUG       = True

# ── Load model once at startup ────────────────────────────────────────
print("Loading model…")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded ✅")

# ── Flask app ─────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow requests from the frontend (index.html)


# ── Helpers ───────────────────────────────────────────────────────────

def preprocess_base64(b64_string: str) -> np.ndarray:
    """
    Accept a base64 data URL (data:image/jpeg;base64,…)
    or a raw base64 string, decode it, resize to 224×224,
    normalise to [0,1], and return shape (1, 224, 224, 3).
    """
    # Strip data URL prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)   # (1, 224, 224, 3)
    return img_array


def hybrid_predict(img_array: np.ndarray, age: int, sex: str,
                   history: str, symptoms: list[str]) -> dict:
    """
    Run the CNN then apply the clinical risk adjustments from
    predict_hybrid.py.  Returns a dict with all result fields.

    NOTE — Label fix:
        Original code had the thresholds inverted
        (risk_score > 0.7 → "BENIGN" is wrong).
        Fixed below: high risk_score → MALIGNANT.
    """
    # ── CNN forward pass ────────────────────────────────────────────
    raw_pred = float(model.predict(img_array, verbose=0)[0][0])

    # ── Risk scoring (mirrors predict_hybrid.py exactly) ────────────
    risk_score = raw_pred

    # Age factor
    if age >= 40:
        risk_score += 0.025
    if age >= 65:
        risk_score += 0.05

    # Sex factor
    if sex.lower() == "male":
        risk_score += 0.02

    # Family history
    if history.lower() == "yes":
        risk_score += 0.1

    # Symptoms — each keyword adds +0.05
    danger_keywords = [
        "bleeding", "itching", "growing",
        "pain", "change", "irregular", "darkening"
    ]
    symptoms_str = " ".join(symptoms).lower()
    for word in danger_keywords:
        if word in symptoms_str:
            risk_score += 0.05

    # Clamp [0, 1]
    risk_score = max(0.0, min(1.0, risk_score))

    # ── Label (FIXED — high score = more dangerous) ──────────────────
    if risk_score >= 0.7:
        label     = "Malignant"
        risk_level = "High"
    elif risk_score >= 0.4:
        label     = "Uncertain"
        risk_level = "Medium"
    else:
        label     = "Benign"
        risk_level = "Low"

    # ── Confidence % (how far from the 0.5 decision boundary) ────────
    confidence = round(abs(risk_score - 0.5) * 2 * 100, 1)   # 0–100
    confidence = max(10.0, min(99.0, confidence))              # clamp

    return {
        "prediction":  label,           # "Benign" | "Malignant" | "Uncertain"
        "riskLevel":   risk_level,       # "Low" | "Medium" | "High"
        "confidence":  confidence,       # float 0–100
        "rawPred":     round(raw_pred, 4),
        "riskScore":   round(risk_score, 4),
    }


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON body:
    {
        "image":   "<base64 data URL or raw base64>",
        "age":     45,
        "sex":     "male" | "female",
        "family":  "yes" | "no",
        "symptoms": ["Bleeding", "Growing", ...]   // array of strings
    }

    Response:
    {
        "prediction":  "Malignant",
        "riskLevel":   "High",
        "confidence":  82.4,
        "rawPred":     0.6821,
        "riskScore":   0.8321,
        "error":       null
    }
    """
    data = request.get_json(force=True)

    # ── Validate required fields ─────────────────────────────────────
    missing = [f for f in ("image", "age", "sex", "family") if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        img_array = preprocess_base64(data["image"])
    except Exception as e:
        return jsonify({"error": f"Image decode failed: {str(e)}"}), 400

    try:
        result = hybrid_predict(
            img_array = img_array,
            age       = int(data["age"]),
            sex       = str(data["sex"]),
            history   = str(data["family"]),
            symptoms  = list(data.get("symptoms", [])),
        )
        result["error"] = None
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)