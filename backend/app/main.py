"""Flask application factory and development server entry point."""
from flask import Flask, jsonify, request, send_from_directory

from .api.routes import api
from .config import FRONTEND_ORIGIN, FRAME_INTERVAL, HOST, MAX_UPLOAD_MB, MAX_VIDEO_SECONDS, MODEL_PATH, PORT, ROOT
from .services.detection_service import DetectionService

FRONTEND_DIR = ROOT / "frontend"


def create_app(detection_service=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    app.config["MAX_VIDEO_SECONDS"] = MAX_VIDEO_SECONDS
    app.extensions["detection_service"] = detection_service or DetectionService(
        MODEL_PATH, FRAME_INTERVAL, MAX_VIDEO_SECONDS
    )
    app.register_blueprint(api)

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

    @app.get("/assets/<path:filename>")
    def frontend_asset(filename):
        return send_from_directory(FRONTEND_DIR / "assets", filename)

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": f"Video exceeds the {MAX_UPLOAD_MB} MB upload limit."}), 413

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
