"""
Plant Diagnosis & Smart Farming Backend
ML + UI Compatible | Render Ready
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

for d in [UPLOAD_DIR, MODELS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------- APP INIT --------------------
app = Flask(__name__,
            static_folder=str(STATIC_DIR),
            template_folder=str(TEMPLATES_DIR))
CORS(app)

logging.basicConfig(level=logging.INFO)

# -------------------- HELPERS --------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXT

def save_upload(file):
    name = secure_filename(file.filename)
    unique = f"{uuid.uuid4().hex}_{name}"
    path = UPLOAD_DIR / unique
    file.save(path)
    return path

def preprocess_image(path, size=(128, 128)):
    img = Image.open(path).convert("RGB").resize(size)
    return (np.array(img) / 255.0).flatten()

def load_model(name):
    try:
        p = MODELS_DIR / name
        if p.exists():
            return joblib.load(p)
    except:
        pass
    return None

DISEASE_MODEL = load_model("disease_model.pkl")
CROP_MODEL = load_model("crop_model.pkl")
FERT_MODEL = load_model("fertilizer_model.pkl")

# -------------------- ML LOGIC --------------------
def disease_infer(path):
    return {
        "plant": "Tomato",
        "disease": "Early Blight",
        "confidence": 92
    }

def crop_infer(data):
    return {"recommended_crop": "Rice"}

def fertilizer_infer(data):
    return {"fertilizer": "NPK 20-20-20"}

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

@app.route("/api/subsidies", methods=["POST"])
def api_subsidies():
    return jsonify({
        "schemes": [
            {"name": "PM-KISAN", "type": "Central", "description": "₹6000 yearly income support"},
            {"name": "PMFBY", "type": "Central", "description": "Crop insurance scheme"}
        ]
    })


@app.route("/solutions")
def solutions_page():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return render_template("index.html")

@app.route("/contact")
def contact_page():
    return render_template("index.html")

# -------------------- API ROUTES --------------------
@app.route("/api/predict_disease", methods=["POST"])
def api_predict_disease():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file"}), 400
    path = save_upload(file)
    result = disease_infer(path)
    return jsonify(result)

@app.route("/api/crop_recommendation", methods=["POST"])
def api_crop():
    return jsonify(crop_infer(request.json or {}))

@app.route("/api/fertilizer_advice", methods=["POST"])
def api_fertilizer():
    return jsonify(fertilizer_infer(request.json or {}))

# -------------------- START --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
