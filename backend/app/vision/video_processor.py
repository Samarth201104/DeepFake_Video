"""Video metadata inspection and deterministic frame sampling."""
import cv2


class VideoReadError(ValueError):
    """The supplied path is not a decodable video."""


VIDEO_BACKENDS = (
    (cv2.CAP_FFMPEG, "FFmpeg"),
    (cv2.CAP_ANY, "system decoder"),
    (cv2.CAP_MSMF, "Media Foundation"),
)


def _open_capture(path):
    """Try Windows' FFmpeg and Media Foundation backends before rejecting a clip."""
    for backend, _name in VIDEO_BACKENDS:
        capture = cv2.VideoCapture(str(path), backend)
        if not capture.isOpened():
            capture.release()
            continue
        ok, first_frame = capture.read()
        if ok and first_frame is not None:
            return capture, backend, first_frame
        capture.release()
    raise VideoReadError(
        "OpenCV could not decode this video. Try converting it to an H.264 MP4 and upload it again."
    )


def inspect_video(path):
    capture, _backend, _first_frame = _open_capture(path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = count / fps if fps and fps > 0 else None
        return duration
    finally:
        capture.release()


def iter_sampled_frames(path, interval):
    capture, _backend, first_frame = _open_capture(path)
    frame_index = 0
    try:
        while True:
            if frame_index == 0:
                ok, frame = True, first_frame
            else:
                ok, frame = capture.read()
            if not ok:
                break
            if frame_index % interval == 0:
                yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()
