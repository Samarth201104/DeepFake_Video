"""Extract RetinaFace crops from fake videos for classifier training."""
import argparse
import random
from pathlib import Path

import cv2
from PIL import Image

from backend.app.vision.face_extractor import extract_faces

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_FACES = 80066
DEFAULT_FRAME_INTERVAL = 15


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "raw" / "fake")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "faces" / "fake")
    parser.add_argument("--frame-interval", type=int, default=DEFAULT_FRAME_INTERVAL)
    parser.add_argument("--target-faces", type=int, default=DEFAULT_TARGET_FACES)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for input video order.")
    return parser.parse_args()


def extract_video_faces(video_path, output_dir, frame_interval):
    """Sample frames and save shared RetinaFace crops as JPEGs."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_index = saved_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_interval == 0:
                for face_index, face in enumerate(extract_faces(frame)):
                    filename = f"{video_path.name}_{frame_index}_{face_index}.jpg"
                    Image.fromarray(face).save(output_dir / filename)
                    saved_count += 1
            frame_index += 1
    finally:
        capture.release()
    return saved_count


def preprocess_fake(input_dir, output_dir, frame_interval, target_faces, seed=None):
    if frame_interval <= 0 or target_faces <= 0:
        raise ValueError("Frame interval and target face count must be positive.")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Fake video directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.jpg"))
    total_faces = len(existing)
    print(f"Resuming with {total_faces} existing fake face crops")
    rng = random.Random(seed)
    videos = [path for path in input_dir.iterdir() if path.is_file()]
    rng.shuffle(videos)
    processed_stems = {
        Path(image.name.rsplit("_", 2)[0]).stem
        for image in existing
        if len(image.name.rsplit("_", 2)) == 3
    }
    total_videos = 0
    for video_path in videos:
        if total_faces >= target_faces:
            break
        if video_path.stem in processed_stems:
            continue
        saved = extract_video_faces(video_path, output_dir, frame_interval)
        total_videos += 1
        total_faces += saved
        print(f"{video_path.name}: {saved} faces saved (total: {total_faces})")
    print(f"Processed {total_videos} new videos; {total_faces} faces total (target: {target_faces})")


def main():
    args = parse_args()
    preprocess_fake(args.input_dir, args.output_dir, args.frame_interval, args.target_faces, args.seed)


if __name__ == "__main__":
    main()
