"""Thread-safe lazy model loader. A checkpoint is loaded once per process."""
from threading import Lock

from tensorflow.keras.models import load_model


class ModelUnavailableError(RuntimeError):
    """Raised when model weights are missing or cannot be loaded."""


_lock = Lock()
_model = None
_model_path = None


def get_model(path):
    global _model, _model_path
    if _model is not None and _model_path == path:
        return _model
    with _lock:
        if _model is not None and _model_path == path:
            return _model
        if not path.is_file():
            raise ModelUnavailableError("Model weights are unavailable. Configure DEEPFAKE_MODEL_PATH.")
        try:
            loaded = load_model(str(path), compile=False)
        except Exception as error:
            raise ModelUnavailableError("The configured model weights could not be loaded.") from error
        _model, _model_path = loaded, path
        return _model
