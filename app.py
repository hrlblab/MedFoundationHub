# app.py
import os, json
from flask import Flask, request, jsonify
from utils import preprocess_image
from inference import generate_report, load_model_config, add_model_from_hf
from PIL import Image
from werkzeug.exceptions import HTTPException

from flask_cors import CORS  # NEW

app = Flask(__name__)
CORS(app, supports_credentials=True)  # NEW

@app.route('/health', methods=['GET'])  # NEW
def health():
    return jsonify({"ok": True})

@app.route('/api/models', methods=['GET'])
def list_models():
    cfg = load_model_config()
    # Include infer_fn (or adapter) so the UI can display it
    return jsonify([
        {
            "name": k,
            "alias": v.get("alias", k),
            "infer_fn": v.get("infer_fn", v.get("adapter", "?"))
        }
        for k, v in cfg.items()
    ])

@app.route('/api/download_model', methods=['POST'])
def download_model():
    req = request.get_json(force=True, silent=True) or {}
    hf_model_id = req.get("hf_model_id")
    local_model_dir = req.get("local_model_dir")
    try:
        if hf_model_id:
            model_keys = add_model_from_hf(hf_model_id)
        elif local_model_dir:
            from inference import add_model_from_local
            model_keys = add_model_from_local(local_model_dir)
        else:
            return jsonify({"success": False, "error": "No model ID or native path specified"})
        return jsonify({"success": True, "model_names": model_keys})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"success": False, "report": None, "error": "No image uploaded"}), 400
    image_file = request.files['image']
    file_ext = image_file.filename.split('.')[-1].lower()
    if file_ext not in ['png', 'tif', 'tiff', 'jpg', 'jpeg']:  # allow JPG
        return jsonify({"success": False, "report": None, "error": "Invalid file type, must be PNG/TIF/TIFF/JPG"}), 400
    user_prompt = request.form.get('prompt', '')
    model_name = request.form.get('model_name', '')
    if not model_name:
        return jsonify({"success": False, "report": None, "error": "No model selected"}), 400
    try:
        image = Image.open(image_file.stream)
        image = preprocess_image(image)
        report = generate_report(image, user_prompt, model_name)
        return jsonify({"success": True, "report": report, "error": None})
    except Exception as e:
        return jsonify({"success": False, "report": None, "error": f"{type(e).__name__}: {str(e)}"}), 500

@app.errorhandler(Exception)  # NEW: nicer unhandled errors
def handle_ex(e):
    if isinstance(e, HTTPException):
        return e
    return jsonify({"success": False, "error": f"ServerError: {str(e)}"}), 500

if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB uploads
    app.run(debug=True, host='0.0.0.0', port=5000)
