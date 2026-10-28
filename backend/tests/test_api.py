"""Fast API route checks that do not require model weights or video fixtures."""
from io import BytesIO
import unittest

from backend.app.main import create_app
from backend.app.services.detection_service import DetectionError


class StubDetectionService:
    def __init__(self):
        self.received_path = None

    def analyze(self, path, filename):
        self.received_path = path
        return {
            "prediction": "Real",
            "confidence": 91.0,
            "fake_score": 9.0,
            "confidence_type": "model score; not a calibrated probability",
            "filename": filename,
            "duration_seconds": 1.0,
            "processed_frames": 1,
            "faces_analyzed": 1,
            "processing_seconds": 0.1,
        }


class MissingModelService:
    def analyze(self, _path, _filename):
        raise DetectionError("Model weights are unavailable.", "model_unavailable")


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.service = StubDetectionService()
        self.app = create_app(self.service)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_health_and_frontend_assets(self):
        health = self.client.get("/api/health")
        page = self.client.get("/")
        script = self.client.get("/assets/js/app.js")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(page.status_code, 200)
        self.assertEqual(script.status_code, 200)
        health.close()
        page.close()
        script.close()

    def test_missing_and_unsupported_uploads(self):
        self.assertEqual(self.client.post("/api/detect").status_code, 400)
        response = self.client.post(
            "/api/detect",
            data={"file": (BytesIO(b"data"), "sample.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 415)

    def test_successful_response_and_temporary_file_cleanup(self):
        response = self.client.post(
            "/api/detect",
            data={"file": (BytesIO(b"video bytes"), "sample.mp4")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["prediction"], "Real")
        self.assertIsNotNone(self.service.received_path)
        self.assertFalse(self.service.received_path.exists())

    def test_malformed_video_is_rejected(self):
        client = create_app().test_client()
        response = client.post(
            "/api/detect",
            data={"file": (BytesIO(b"not a video"), "broken.mp4")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("readable video", response.json["error"])
        response.close()

    def test_missing_model_returns_service_unavailable(self):
        client = create_app(MissingModelService()).test_client()
        response = client.post(
            "/api/detect",
            data={"file": (BytesIO(b"video"), "sample.mp4")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 503)
        response.close()


if __name__ == "__main__":
    unittest.main()
