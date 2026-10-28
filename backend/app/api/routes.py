"""JSON endpoints for health checks and video analysis."""
from flask import Blueprint, current_app, jsonify, request
from pathlib import Path
import tempfile

from ..config import ALLOWED_EXTENSIONS, MAX_UPLOAD_MB, MODEL_PATH
from ..services.detection_service import DetectionError

api = Blueprint("api", __name__, url_prefix="/api")


@api.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_configured": MODEL_PATH.is_file(),
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_duration_seconds": current_app.config["MAX_VIDEO_SECONDS"],
        "supported_formats": sorted(ALLOWED_EXTENSIONS),
    })


@api.post("/detect")
def detect():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose a video file to analyze."}), 400
    filename = Path(upload.filename).name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported video format."}), 415

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="deepfake_", suffix=suffix, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            upload.save(temporary)
        if temporary_path.stat().st_size == 0:
            return jsonify({"error": "The uploaded file is empty."}), 400
        result = current_app.extensions["detection_service"].analyze(temporary_path, filename)
        return jsonify(result)
    except DetectionError as error:
        status = 503 if error.code == "model_unavailable" else 422
        return jsonify({"error": str(error)}), status
    except Exception:
        current_app.logger.exception("Video inference failed")
        return jsonify({"error": "Video analysis failed. Check the file and try again."}), 500
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                current_app.logger.warning("Could not remove temporary upload")
