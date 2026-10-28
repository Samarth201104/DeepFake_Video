"""Orchestrates the preserved RetinaFace + Xception video detection pipeline."""
import time

import numpy as np

from ..models.model_loader import ModelUnavailableError, get_model
from ..vision.face_extractor import extract_faces
from ..vision.preprocessing import normalize_image
from ..vision.video_processor import VideoReadError, inspect_video, iter_sampled_frames


class DetectionError(RuntimeError):
    def __init__(self, message, code="invalid_video"):
        super().__init__(message)
        self.code = code


class DetectionService:
    def __init__(self, model_path, frame_interval=10, max_video_seconds=600):
        self.model_path = model_path
        self.frame_interval = frame_interval
        self.max_video_seconds = max_video_seconds

    def analyze(self, path, filename):
        try:
            duration = inspect_video(path)
        except VideoReadError as error:
            raise DetectionError(str(error)) from error
        if duration is not None and duration > self.max_video_seconds:
            raise DetectionError("Video exceeds the configured duration limit.")
        try:
            model = get_model(self.model_path)
        except ModelUnavailableError as error:
            raise DetectionError(str(error), "model_unavailable") from error

        started = time.perf_counter()
        probabilities = []
        sampled_count = 0
        faces_count = 0
        try:
            for _, frame in iter_sampled_frames(path, self.frame_interval):
                sampled_count += 1
                for face in extract_faces(frame):
                    batch = np.expand_dims(normalize_image(face), axis=0)
                    score = float(np.asarray(model.predict(batch, verbose=0)).reshape(-1)[0])
                    if np.isfinite(score):
                        probabilities.append(float(np.clip(score, 0.0, 1.0)))
                        faces_count += 1
        except VideoReadError as error:
            raise DetectionError(str(error)) from error
        if not probabilities:
            raise DetectionError("No faces were detected in the sampled video frames.")

        fake_score = float(np.mean(probabilities))
        return {
            "prediction": "Fake" if fake_score >= 0.5 else "Real",
            "confidence": round(max(fake_score, 1 - fake_score) * 100, 2),
            "fake_score": round(fake_score * 100, 2),
            "confidence_type": "model score; not a calibrated probability",
            "filename": filename,
            "duration_seconds": round(duration, 2) if duration is not None else None,
            "processed_frames": sampled_count,
            "faces_analyzed": faces_count,
            "processing_seconds": round(time.perf_counter() - started, 2),
        }
