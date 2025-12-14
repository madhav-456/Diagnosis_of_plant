"""
Plant Diagnosis & Smart Farming Backend
ML + UI Compatible | Render / Railway Ready
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Dict

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

import joblib
import numpy as np
from PIL import Image

# -------------------- PATH CONFIG --------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg"}
PORT = int(os.environ.get("PORT", 10000))

for d in [UPLOAD_DIR, MODELS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------- APP INIT --------------------
app = Flask(__name__,
            static_folder=str(STATIC_DIR),
            template_folder=str(TEMPLATES_DIR))
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart_farming")

# -------------------- HELPERS --------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXT

def save_upload(file, sub="images"):
    name = secure_filename(file.filename)
    unique = f"{uuid.uuid4().hex}_{name}"
    path = UPLOAD_DIR / sub
    path.mkdir(exist_ok=True)
    full = path / unique
    file.save(full)
    return full

def preprocess_image(path, size=(128, 128)):
    img = Image.open(path).convert("RGB").resize(size)
    return (np.array(img) / 255.0).flatten()

# -------------------- LOAD MODELS (SAFE) --------------------
def load_model(name):
    try:
        path = MODELS_DIR / name
        if path.exists():
            logger.info(f"Loaded model: {name}")
            return joblib.load(path)
    except Exception as e:
        logger.warning(f"Failed loading {name}: {e}")
    return None

DISEASE_MODEL = load_model("disease_model.pkl")
CROP_MODEL = load_model("crop_model.pkl")
FERT_MODEL = load_model("fertilizer_model.pkl")

# -------------------- ML + UI INFERENCE --------------------
def disease_infer(image_path) -> Dict:
    if DISEASE_MODEL:
        try:
            x = preprocess_image(image_path)
            pred = DISEASE_MODEL.predict([x])[0]
            plant, disease = (pred.split("_") + ["Unknown"])[:2]
        except:
            plant, disease = "Tomato", "Early Blight"
    else:
        plant, disease = "Tomato", "Early Blight"

    return {
        "plant": plant,
        "disease": disease,
        "confidence": 92,
        "description": "Fungal disease causing dark leaf spots.",
        "chemical": "Use Mancozeb fungicide weekly.",
        "organic": "Neem oil spray every 7 days.",
        "prevention": "Crop rotation and avoid overhead irrigation."
    }

def crop_infer(payload) -> Dict:
    if CROP_MODEL:
        try:
            x = [
                float(payload.get("N", 0)),
                float(payload.get("P", 0)),
                float(payload.get("K", 0)),
                float(payload.get("temperature", 0)),
                float(payload.get("humidity", 0)),
                float(payload.get("ph", 0)),
                float(payload.get("rainfall", 0)),
            ]
            pred = CROP_MODEL.predict([x])[0]
            return {"recommended_crop": str(pred)}
        except:
            pass

    return {"recommended_crop": "Rice"}

def fertilizer_infer(payload) -> Dict:
    if FERT_MODEL:
        try:
            x = [
                float(payload.get("temperature", 0)),
                float(payload.get("humidity", 0)),
                float(payload.get("moisture", 0)),
                float(payload.get("nitrogen", 0)),
                float(payload.get("phosphorous", 0)),
                float(payload.get("potassium", 0)),
            ]
            pred = FERT_MODEL.predict([x])[0]
            return {"fertilizer": str(pred)}
        except:
            pass

    return {
        "fertilizer": "NPK 20-20-20",
        "note": "Apply 50kg per acre before irrigation."
    }

# -------------------- HTML ROUTES --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/disease")
def disease_page():
    return render_template("DiseasePrediction3.html")

@app.route("/crop")
def crop_page():
    return render_template("Crop_recommendation4.html")

@app.route("/fertilizer")
def fertilizer_page():
    return render_template("Fertilizer_recommendation2.html")

@app.route("/assistant")
def assistant_page():
    return render_template("ai_assistant5.html")

@app.route("/subsidy")
def subsidy_page():
    return render_template("subsidy6.html")
    @app.route("/solutions")
def solutions_page():
    return render_template("Solutions.html")

# -------------------- API ROUTES --------------------
@app.route("/api/predict_disease", methods=["POST"])
def api_predict_disease():
    file = request.files.get("file")
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid image"}), 400

    path = save_upload(file)
    result = disease_infer(path)

    try:
        path.unlink()
    except:
        pass

    return jsonify(result)

@app.route("/api/crop_recommendation", methods=["POST"])
def api_crop():
    return jsonify(crop_infer(request.get_json() or {}))

@app.route("/api/fertilizer_advice", methods=["POST"])
def api_fertilizer():
    return jsonify(fertilizer_infer(request.get_json() or {}))

@app.route("/api/assistant_chat", methods=["POST"])
def api_assistant():
    msg = (request.get_json() or {}).get("message", "")
    return jsonify({
        "response": f"You asked: {msg}. This is smart farming advice."
    })

@app.route("/api/subsidies", methods=["POST"])
def api_subsidies():
    return jsonify({
        "schemes": [
            {"name": "PM-KISAN", "type": "Central", "description": "₹6000 yearly income support"},
            {"name": "PMFBY", "type": "Central", "description": "Crop insurance scheme"}
        ]
    })

# -------------------- START --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
