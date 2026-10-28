"""RetinaFace wrapper shared by dataset preparation and video inference."""
import cv2
import numpy as np
from retinaface import RetinaFace

from .preprocessing import resize_face


def extract_faces(frame_bgr, target_size=(256, 256)):
    try:
        detections = RetinaFace.detect_faces(frame_bgr)
    except Exception:
        return []
    if not isinstance(detections, dict):
        return []
    height, width = frame_bgr.shape[:2]
    output = []
    for detection in detections.values():
        x1, y1, x2, y2 = detection["facial_area"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size:
            output.append(resize_face(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), target_size))
    return output
