"""
Plant Diagnosis & Smart Farming Backend (single-file Flask app)

Features:
 - Voice Assistant (audio upload -> optional auto transcription if speech_recognition + pocketsphinx installed)
 - Text AI assistant (demo; hook to call real LLM)
 - Crop disease detection via image upload (uses sklearn/ joblib .pkl models if present)
 - Crop recommendation and fertilizer recommendation (JSON inputs -> model or fallback)
 - Subsidy finder (GET or POST)
 - Serves pages (templates) and static assets

Drop into your repo root next to: models/, static/, templates/
"""

import os
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
import joblib
from PIL import Image
import numpy as np
import io

# Optional speech libs (used only if installed)
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# -------------------- Configuration --------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif"}
ALLOWED_AUDIO_EXT = {"wav", "mp3", "m4a", "ogg"}
MAX_UPLOAD_BYTES = 6 * 1024 * 1024  # 6 MB

for d in [UPLOAD_DIR, MODELS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------- Flask app --------------------
app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATES_DIR),
)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
PORT = int(os.environ.get("PORT", 10000))

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("smart_farming")

# -------------------- Helpers --------------------
def allowed_file(filename: str, allowed: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed

def save_upload(file_storage, subfolder: Optional[Path] = None) -> Path:
    filename = secure_filename(file_storage.filename)
    unique = f"{uuid.uuid4().hex}_{filename}"
    dest_dir = UPLOAD_DIR if subfolder is None else UPLOAD_DIR / subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)
    save_path = dest_dir / unique
    file_storage.save(str(save_path))
    logger.info("Saved upload to %s", save_path)
    return save_path

def preprocess_image(image_path: Path, size=(128, 128)) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    arr = np.asarray(img) / 255.0
    return arr.flatten()

# -------------------- Model Loading --------------------
def load_model(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        logger.warning("Model file not found: %s", path)
        return None
    try:
        m = joblib.load(path)
        logger.info("Loaded model: %s", path)
        return m
    except Exception as e:
        logger.exception("Failed to load model %s: %s", path, e)
        return None

DISEASE_MODEL = load_model("disease_model.pkl")
FERT_MODEL = load_model("fertilizer_model.pkl")
CROP_MODEL = load_model("crop_model.pkl")

# -------------------- AI Assistant --------------------
def _call_openai(prompt: str) -> Optional[str]:
    try:
        import openai
    except Exception:
        return None
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    openai.api_key = key
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            max_tokens=512,
            temperature=float(os.getenv("OPENAI_TEMPERATURE","0.2"))
        )
        if hasattr(resp, "choices") and len(resp.choices)>0:
            text = resp.choices[0].message.get("content") if isinstance(resp.choices[0].message, dict) else resp.choices[0].message
            return text or str(resp)
        return str(resp)
    except Exception:
        return None

def ai_assistant_query(prompt: str) -> Dict[str, Any]:
    providers = os.getenv("LLM_PROVIDER_ORDER", "OPENAI").split(",")
    providers = [p.strip().upper() for p in providers if p.strip()]
    result_text = None
    for p in providers:
        if p=="OPENAI":
            result_text = _call_openai(prompt)
        if result_text:
            return {"provider": p, "query": prompt, "answer": result_text}
    return {"provider": "DEMO", "query": prompt, "answer": "Demo AI response. Connect a real LLM in ai_assistant_query()."}

# -------------------- Inference / Business Logic --------------------
def disease_infer(image_path: Path) -> Dict[str, Any]:
    if DISEASE_MODEL is not None:
        try:
            x = preprocess_image(image_path)
            pred = DISEASE_MODEL.predict([x])[0]
            prob = None
            if hasattr(DISEASE_MODEL, "predict_proba"):
                try:
                    prob = float(DISEASE_MODEL.predict_proba([x]).max())
                except Exception:
                    prob = None
            # AI-enhanced explanation
            prompt = f"A plant leaf image was analyzed. Disease: {pred}. Explain causes, symptoms, treatment, prevention."
            ai_resp = ai_assistant_query(prompt)
            return {"plant": str(pred.split("_")[0] if "_" in pred else pred),
                    "disease": str(pred.split("_")[1] if "_" in pred else pred),
                    "confidence": prob,
                    "ai_explanation": ai_resp.get("answer"),
                    "ai_provider": ai_resp.get("provider")}
        except Exception as e:
            logger.exception("Disease inference error: %s", e)
    return {
        "plant": "Tomato",
        "disease": "Early Blight",
        "confidence": 0.87,
        "ai_explanation": "Demo explanation. Connect LLM for real explanation.",
        "ai_provider": "DEMO"
    }

def fertilizer_infer(payload: dict) -> Dict[str, Any]:
    if FERT_MODEL is not None:
        try:
            features = [
                float(payload.get("soil_ph", 0)),
                float(payload.get("nitrogen", 0)),
                float(payload.get("phosphorus", 0)),
                float(payload.get("potassium", 0)),
            ]
            pred = FERT_MODEL.predict([features])[0]
            return {"recommendation": str(pred)}
        except Exception as e:
            logger.exception("Fertilizer model error: %s", e)
    crop = payload.get("crop", "unknown")
    return {"recommendation": f"Use balanced NPK for {crop} (demo advice)"}

def crop_infer(payload: dict) -> Dict[str, Any]:
    if CROP_MODEL is not None:
        try:
            features = [
                float(payload.get("soil_ph", 0)),
                float(payload.get("rainfall", 0)),
                float(payload.get("temperature", 0)),
            ]
            pred = CROP_MODEL.predict([features])[0]
            return {"recommended_crop": str(pred)}
        except Exception as e:
            logger.exception("Crop model error: %s", e)
    return {"recommended_crop": "Maize (demo)"}

def subsidy_lookup(state: str = "IN", crop: Optional[str] = None, land_size: Optional[float] = None) -> Dict[str, Any]:
    subs = [
        {"program": "GreenGrow", "amount": "20000", "currency": "INR", "eligibility": "Smallholders"},
        {"program": "Irrigation Grant", "amount": "50000", "currency": "INR", "eligibility": "Irrigation Systems"}
    ]
    if land_size and float(land_size) > 5:
        subs.append({"program": "Large Farm Support", "amount": "100000", "currency": "INR", "eligibility": "Large farms (>5 acres)"})
    return {"state": state, "crop": crop, "land_size": land_size, "subsidies": subs}

# -------------------- Voice Utilities --------------------
def transcribe_audio_file(path: Path) -> str:
    if not SR_AVAILABLE:
        return "Transcription unavailable (speech_recognition not installed)."
    try:
        r = sr.Recognizer()
        with sr.AudioFile(str(path)) as source:
            audio = r.record(source)
        try:
            return r.recognize_sphinx(audio)
        except Exception:
            return r.recognize_google(audio)
    except Exception:
        return "Could not transcribe audio."

def synthesize_text_to_audio(text: str) -> Optional[bytes]:
    if not TTS_AVAILABLE:
        return None
    try:
        engine = pyttsx3.init()
        tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_tts.wav"
        engine.save_to_file(text, str(tmp_path))
        engine.runAndWait()
        with open(tmp_path, "rb") as f:
            data = f.read()
        tmp_path.unlink(missing_ok=True)
        return data
    except Exception:
        return None

# -------------------- Routes --------------------
@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception:
        return "<h1>Plant Diagnosis & Smart Farming API</h1><p>Use the /api endpoints.</p>"

@app.route("/landing")
def landing():
    try:
        return render_template("landingpage1.html")
    except Exception:
        return "<h1>Landing</h1>"

# Static file routes
@app.route("/image/<path:filename>")
def image_files(filename):
    return send_from_directory(STATIC_DIR / "images", filename)

@app.route("/css/<path:filename>")
def css_files(filename):
    return send_from_directory(STATIC_DIR / "css", filename)

@app.route("/js/<path:filename>")
def js_files(filename):
    return send_from_directory(STATIC_DIR / "js", filename)

# -------------------- API Endpoints --------------------
@app.route("/api/predict_disease", methods=["POST"])
def api_predict_disease():
    if "file" not in request.files:
        return jsonify({"error": "file required"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400
    if not allowed_file(file.filename, ALLOWED_IMAGE_EXT):
        return jsonify({"error": f"invalid file type. allowed: {sorted(ALLOWED_IMAGE_EXT)}"}), 400
    try:
        saved = save_upload(file, subfolder=Path("images"))
        result = disease_infer(saved)
        try: saved.unlink(missing_ok=True)
        except Exception: pass
        return jsonify({"status": "success", "result": result})
    except Exception:
        return jsonify({"error": "internal server error"}), 500

@app.route("/api/fertilizer_advice", methods=["POST"])
def api_fertilizer():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON required"}), 400
    result = fertilizer_infer(payload)
    return jsonify({"status": "success", "result": result})

@app.route("/api/crop_recommendation", methods=["POST"])
def api_crop():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON required"}), 400
    data = {"soil_ph": payload.get("ph", payload.get("soil_ph", 0)),
            "rainfall": payload.get("rainfall", 0),
            "temperature": payload.get("temperature", 0)}
    result = crop_infer(data)
    return jsonify({"status": "success", "result": result})

@app.route("/api/ai_assistant", methods=["POST"])
def api_ai():
    payload = request.get_json(silent=True)
    if not payload or "query" not in payload:
        return jsonify({"error": "query required"}), 400
    result = ai_assistant_query(payload["query"])
    return jsonify({"status": "success", "result": result})

@app.route("/api/voice_transcribe", methods=["POST"])
def api_voice_transcribe():
    if "file" not in request.files:
        return jsonify({"error": "file required"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400
    if not allowed_file(file.filename, ALLOWED_AUDIO_EXT):
        return jsonify({"error": f"invalid audio type. allowed: {sorted(ALLOWED_AUDIO_EXT)}"}), 400
    saved = save_upload(file, subfolder=Path("audio"))
    text = transcribe_audio_file(saved)
    try: saved.unlink(missing_ok=True)
    except Exception: pass
    return jsonify({"status": "success", "transcription": text})

@app.route("/api/text_to_speech", methods=["POST"])
def api_text_to_speech():
    payload = request.get_json(silent=True)
    if not payload or "text" not in payload:
        return jsonify({"error": "text required"}), 400
    audio_bytes = synthesize_text_to_audio(payload["text"])
    if audio_bytes is None:
        return jsonify({"status": "success", "message": "TTS not available on this server. Install pyttsx3."})
    return send_file(io.BytesIO(audio_bytes), mimetype="audio/wav", as_attachment=False, download_name="speech.wav")

@app.route("/api/subsidies", methods=["GET", "POST"])
def api_subsidies():
    if request.method == "GET":
        state = request.args.get("state", "IN")
        crop = request.args.get("crop")
        land_size = request.args.get("land_size")
    else:
        payload = request.get_json(silent=True) or {}
        state = payload.get("state", "IN")
        crop = payload.get("crop")
        land_size = payload.get("land_size")
    result = subsidy_lookup(state, crop, land_size)
    return jsonify({"status": "success", "result": result})

# -------------------- Error handlers --------------------
@app.errorhandler(404)
def not_found(err):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(413)
def payload_too_large(err):
    return jsonify({"error": f"Uploaded file too large. Max {MAX_UPLOAD_BYTES} bytes."}), 413

# -------------------- Run --------------------
if __name__ == "__main__":
    logger.info("Starting Smart Farming backend on port %s", PORT)
    app.run(host="0.0.0.0", port=PORT, debug=False)
