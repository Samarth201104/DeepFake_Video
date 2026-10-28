"""Model lifecycle and video inference; preserves the repository's trained pipeline."""
import time

import cv2
import numpy as np
from PIL import Image
from retinaface import RetinaFace
from tensorflow.keras.models import load_model


class InferenceError(Exception):
    """Expected, safe-to-report inference failure."""


class Detector:
    def __init__(self, model_path, frame_interval=10, max_video_seconds=600):
        self.model_path = model_path
        self.frame_interval = frame_interval
        self.max_video_seconds = max_video_seconds
        self.model = None

    def load(self):
        if self.model is not None:
            return self.model
        if not self.model_path.is_file():
            raise InferenceError("Model weights are unavailable. Configure DEEPFAKE_MODEL_PATH.")
        try:
            self.model = load_model(str(self.model_path), compile=False)
        except Exception as exc:
            raise InferenceError("The configured model weights could not be loaded.") from exc
        return self.model

    @staticmethod
    def extract_faces(frame, target_size=(256, 256)):
        try:
            detections = RetinaFace.detect_faces(frame)
        except Exception:
            return []
        if not isinstance(detections, dict):
            return []
        height, width = frame.shape[:2]
        faces = []
        for item in detections.values():
            x1, y1, x2, y2 = item["facial_area"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                faces.append(np.asarray(Image.fromarray(rgb).resize(target_size)))
        return faces

    def analyze(self, path, filename):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise InferenceError("The uploaded file is not a readable video.")
        model = self.load()
        started = time.perf_counter()
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps and fps > 0 else None
        if duration is not None and duration > self.max_video_seconds:
            capture.release()
            raise InferenceError("Video exceeds the configured duration limit.")
        probabilities = []
        faces_count = 0
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % self.frame_interval == 0:
                    faces = self.extract_faces(frame)
                    for face in faces:
                        batch = np.expand_dims(face.astype(np.float32) / 255.0, axis=0)
                        score = float(np.asarray(model.predict(batch, verbose=0)).reshape(-1)[0])
                        if not np.isfinite(score):
                            continue
                        probabilities.append(float(np.clip(score, 0.0, 1.0)))
                        faces_count += 1
                frame_index += 1
        finally:
            capture.release()
        if not probabilities:
            raise InferenceError("No faces were detected in the sampled video frames.")
        fake_score = float(np.mean(probabilities))
        label = "Fake" if fake_score >= 0.5 else "Real"
        return {
            "prediction": label,
            "confidence": round(max(fake_score, 1 - fake_score) * 100, 2),
            "fake_score": round(fake_score * 100, 2),
            "confidence_type": "model score; not a calibrated probability",
            "filename": filename,
            "duration_seconds": round(duration, 2) if duration is not None else None,
            "processed_frames": int(np.ceil(frame_index / self.frame_interval)),
            "faces_analyzed": faces_count,
            "processing_seconds": round(time.perf_counter() - started, 2),
        }
