"""Centralized backend settings. Values can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(
    os.getenv("DEEPFAKE_MODEL_PATH", str(ROOT / "artifacts" / "models" / "xception_best.h5")).strip()
).expanduser()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "600"))
FRAME_INTERVAL = int(os.getenv("FRAME_INTERVAL", "10"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://127.0.0.1:5000")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "5000"))
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
