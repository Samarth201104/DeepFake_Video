"""Video metadata inspection and deterministic frame sampling."""
import cv2


class VideoReadError(ValueError):
    """The supplied path is not a decodable video."""


def inspect_video(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoReadError("The uploaded file is not a readable video.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = count / fps if fps and fps > 0 else None
        return duration
    finally:
        capture.release()


def iter_sampled_frames(path, interval):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoReadError("The uploaded file is not a readable video.")
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % interval == 0:
                yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()
