"""
Plant Diagnosis & Smart Farming Backend (cleaned & Render-ready)
"""

import os
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import joblib
from PIL import Image
import numpy as np

# -------------------- CONFIG --------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg"}
MAX_UPLOAD_BYTES = 6 * 1024 * 1024  # 6MB

for d in [UPLOAD_DIR, MODELS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATES_DIR),
)
CORS(app)

PORT = int(os.environ.get("PORT", 10000))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("smart_farming")

# -------------------- HELPERS --------------------
def allowed_file(filename: str, allowed: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed

def save_upload(file_storage, subfolder=None) -> Path:
    filename = secure_filename(file_storage.filename)
    unique = f"{uuid.uuid4().hex}_{filename}"
    dest_dir = UPLOAD_DIR if subfolder is None else UPLOAD_DIR / subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)
    save_path = dest_dir / unique
    file_storage.save(str(save_path))
    return save_path

def preprocess_image(image_path: Path, size=(128, 128)) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    return (np.asarray(img) / 255.0).flatten()

# -------------------- MODEL LOADING --------------------
def load_model(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        logger.warning(f"Missing model: {filename}")
        return None
    try:
        m = joblib.load(path)
        logger.info(f"Loaded model: {filename}")
        return m
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        return None

DISEASE_MODEL = load_model("disease_model.pkl")
CROP_MODEL = load_model("crop_model.pkl")
CROP_ENCODER = load_model("crop_encoder.pkl")
FERT_MODEL = load_model("fertilizer_model.pkl")
FERT_ENCODER = load_model("fertilizer_encoder.pkl")
SOIL_ENCODER = load_model("soil_encoder.pkl")

# -------------------- INFERENCE --------------------
def disease_infer(image_path: Path) -> Dict:
    if DISEASE_MODEL:
        try:
            x = preprocess_image(image_path)
            pred = DISEASE_MODEL.predict([x])[0]
            plant, disease = (pred.split("_") + ["Unknown"])[:2]

            return {
                "plant": plant,
                "disease": disease,
                "confidence": None,
                "ai_explanation": "Upload OpenAI key for full explanation",
                "ai_provider": "NONE"
            }
        except Exception as e:
            logger.error(f"Disease error: {e}")

    return {
        "plant": "Tomato",
        "disease": "Early Blight",
        "confidence": 0.8,
        "ai_explanation": "Demo response",
        "ai_provider": "DEMO"
    }

def crop_infer(payload: dict) -> Dict:
    if CROP_MODEL:
        try:
            x = [
                float(payload.get("soil_ph", 0)),
                float(payload.get("rainfall", 0)),
                float(payload.get("temperature", 0)),
            ]
            pred = CROP_MODEL.predict([x])[0]
            return {"recommended_crop": str(pred)}
        except Exception as e:
            logger.error(f"Crop error: {e}")

    return {"recommended_crop": "Maize (demo)"}

def fertilizer_infer(payload: dict) -> Dict:
    if FERT_MODEL:
        try:
            x = [
                float(payload.get("soil_ph", 0)),
                float(payload.get("nitrogen", 0)),
                float(payload.get("phosphorus", 0)),
                float(payload.get("potassium", 0)),
            ]
            pred = FERT_MODEL.predict([x])[0]
            return {"recommendation": str(pred)}
        except Exception as e:
            logger.error(f"Fertilizer error: {e}")

    return {"recommendation": "Use NPK (demo)"}

# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return "<h1>Plant Diagnosis API Running</h1>"

@app.route("/api/predict_disease", methods=["POST"])
def api_predict_disease():
    if "file" not in request.files:
        return jsonify({"error": "file required"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename, ALLOWED_IMAGE_EXT):
        return jsonify({"error": "Invalid image type"}), 400

    path = save_upload(file, "images")
    result = disease_infer(path)

    try:
        path.unlink()
    except:
        pass

    return jsonify({"status": "success", "result": result})

@app.route("/api/crop_recommendation", methods=["POST"])
def api_crop():
    result = crop_infer(request.get_json() or {})
    return jsonify({"status": "success", "result": result})

@app.route("/api/fertilizer_advice", methods=["POST"])
def api_fert():
    result = fertilizer_infer(request.get_json() or {})
    return jsonify({"status": "success", "result": result})

# -------------------- START --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

