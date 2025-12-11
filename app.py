import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import uuid
import joblib
from PIL import Image
import numpy as np

# ------------------ Base Directories ------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"

ALLOWED_EXT = {"png", "jpg", "jpeg", "gif"}

# Create folders safely (handles existing files too)
for folder in [UPLOAD_DIR, MODELS_DIR]:
    if folder.exists():
        if not folder.is_dir():
            # If a file exists with the same name, remove it first
            folder.unlink()
            folder.mkdir(parents=True, exist_ok=True)
    else:
        folder.mkdir(parents=True, exist_ok=True)

# ------------------ Flask App ------------------
app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder="templates",
    static_url_path="/static"
)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB
CORS(app)

# ------------------ Helpers ------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# ------------------ Model Loader ------------------
def load_model(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        print(f"Model {filename} not found in {MODELS_DIR}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Model load error ({filename}): {e}")
        return None

DISEASE_MODEL = load_model("disease_model.pkl")
FERT_MODEL = load_model("fertilizer_model.pkl")
CROP_MODEL = load_model("crop_model.pkl")

# ------------------ Prediction Helpers ------------------
def preprocess_image_for_model(image_path: str, size=(128, 128)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    arr = np.asarray(img) / 255.0
    return arr.flatten()

def disease_infer(image_path: str):
    if DISEASE_MODEL:
        try:
            x = preprocess_image_for_model(image_path)
            pred = DISEASE_MODEL.predict([x])[0]
            prob = float(DISEASE_MODEL.predict_proba([x]).max()) if hasattr(DISEASE_MODEL, "predict_proba") else None
            return {"plant": "unknown", "disease": str(pred), "confidence": prob}
        except Exception as e:
            print("Disease model error:", e)
    return {"plant": "Tomato", "disease": "Early blight (dummy)", "confidence": 0.87,
            "advice": ["Remove infected leaves", "Use copper-based fungicide", "Improve ventilation"]}

def fertilizer_infer(payload: dict):
    if FERT_MODEL:
        try:
            features = [payload.get("soil_ph", 0), payload.get("nitrogen", 0),
                        payload.get("phosphorus", 0), payload.get("potassium", 0)]
            pred = FERT_MODEL.predict([features])[0]
            return {"recommendation": str(pred)}
        except Exception as e:
            print("Fertilizer model error:", e)
    crop = payload.get("crop", "unknown")
    return {"recommendation": f"Use balanced NPK for {crop} (dummy advice)"}

def crop_infer(payload: dict):
    if CROP_MODEL:
        try:
            features = [payload.get("soil_ph", 0), payload.get("rainfall", 0), payload.get("temperature", 0)]
            pred = CROP_MODEL.predict([features])[0]
            return {"recommended_crop": str(pred)}
        except Exception as e:
            print("Crop model error:", e)
    return {"recommended_crop": "Maize"}

def ai_assistant_query(q: str):
    return {"query": q, "answer": "Demo AI response. Connect real LLM later."}

def subsidy_lookup(country="IN", crop=None):
    return {"country": country, "crop": crop, "subsidies": [
        {"program": "GreenGrow", "amount": "20000", "currency": "INR", "eligibility": "Smallholders"},
        {"program": "Irrigation Grant", "amount": "50000", "currency": "INR", "eligibility": "Irrigation Systems"}
    ]}

# ------------------ HTML Pages ------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/landing")
def landing():
    return render_template("landingpage1.html")

@app.route("/DiseasePrediction3")
def disease_page():
    return render_template("DiseasePrediction3.html")

@app.route("/Fertilizer_recommendation2")
def fert_page():
    return render_template("Fertilizer_recommendation2.html")

@app.route("/Crop_recommendation4")
def crop_page():
    return render_template("Crop_recommendation4.html")

@app.route("/ai_assistant5")
def ai_page():
    return render_template("ai_assistant5.html")

@app.route("/subsidy6")
def subsidy_page():
    return render_template("subsidy6.html")

# ------------------ Static Files ------------------
@app.route("/image/<path:filename>")
def image_files(filename):
    return send_from_directory(STATIC_DIR / "image", filename)

@app.route("/css/<path:filename>")
def css_files(filename):
    return send_from_directory(STATIC_DIR / "css", filename)

@app.route("/js/<path:filename>")
def js_files(filename):
    return send_from_directory(STATIC_DIR / "js", filename)

# ------------------ API ------------------
@app.route("/api/predict_disease", methods=["POST"])
def api_predict_disease():
    if "file" not in request.files:
        return jsonify({"error": "file required"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "invalid file type"}), 400
    fname = secure_filename(file.filename)
    unique = f"{uuid.uuid4().hex}_{fname}"
    save_path = UPLOAD_DIR / unique
    file.save(str(save_path))
    result = disease_infer(str(save_path))
    return jsonify({"status": "success", "result": result})

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
    data = {"soil_ph": payload.get("ph", 0), "rainfall": payload.get("rainfall", 0), "temperature": payload.get("temperature", 0)}
    result = crop_infer(data)
    return jsonify(result)

@app.route("/api/ai_assistant", methods=["POST"])
def api_ai():
    payload = request.get_json(silent=True)
    if not payload or "query" not in payload:
        return jsonify({"error": "query required"}), 400
    result = ai_assistant_query(payload["query"])
    return jsonify({"status": "success", "result": result})

@app.route("/api/subsidies", methods=["GET"])
def api_subsidies():
    country = request.args.get("country", "IN")
    crop = request.args.get("crop")
    result = subsidy_lookup(country, crop)
    return jsonify({"status": "success", "result": result})

# ------------------ Error ------------------
@app.errorhandler(404)
def not_found(err):
    return jsonify({"error": "Not found"}), 404

# ------------------ Run App ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render uses $PORT
    app.run(host="0.0.0.0", port=port)

