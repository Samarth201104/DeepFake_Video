"""HTTP API and frontend entry point."""
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .config import ALLOWED_EXTENSIONS, FRAME_INTERVAL, FRONTEND_ORIGIN, MAX_UPLOAD_MB, MAX_VIDEO_SECONDS, MODEL_PATH, ROOT
from .inference import Detector, InferenceError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
detector = Detector(MODEL_PATH, FRAME_INTERVAL, MAX_VIDEO_SECONDS)
FRONTEND_DIR = ROOT / "frontend"


@app.after_request
def cors(response):
    origin = request.headers.get("Origin")
    if origin == FRONTEND_ORIGIN:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/frontend/<path:filename>")
def frontend_asset(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_configured": MODEL_PATH.is_file(), "max_upload_mb": MAX_UPLOAD_MB,
                    "max_duration_seconds": MAX_VIDEO_SECONDS, "supported_formats": sorted(ALLOWED_EXTENSIONS)})


@app.post("/api/detect")
def detect():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose a video file to analyze."}), 400
    filename = Path(upload.filename).name
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported video format."}), 415
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="deepfake_", suffix=Path(filename).suffix.lower(), delete=False) as temp:
            temp_path = Path(temp.name)
            upload.save(temp)
        if temp_path.stat().st_size == 0:
            return jsonify({"error": "The uploaded file is empty."}), 400
        result = detector.analyze(temp_path, filename)
        return jsonify(result)
    except InferenceError as exc:
        status = 503 if "weights" in str(exc).lower() or "model" in str(exc).lower() else 422
        return jsonify({"error": str(exc)}), status
    except Exception:
        app.logger.exception("Video inference failed")
        return jsonify({"error": "Video analysis failed. Check the file and try again."}), 500
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                app.logger.warning("Could not remove temporary upload")


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": f"Video exceeds the {MAX_UPLOAD_MB} MB upload limit."}), 413


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False)
