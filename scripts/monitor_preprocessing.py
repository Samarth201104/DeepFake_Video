"""Periodically report fake-video preprocessing progress."""
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "raw" / "fake"
OUTPUT_DIR = ROOT / "data" / "processed" / "faces" / "fake"


def count_progress():
    images = list(OUTPUT_DIR.glob("*.jpg")) if OUTPUT_DIR.exists() else []
    names = {"_".join(image.name.split("_")[:-2]) for image in images}
    video_total = sum(path.is_file() for path in DATASET_DIR.iterdir()) if DATASET_DIR.exists() else 0
    return len(names), len(images), video_total


def main():
    try:
        while True:
            videos_done, frames, videos_total = count_progress()
            print(f"Fake videos: {videos_done}/{videos_total} | Faces extracted: {frames}")
            print("-" * 50)
            time.sleep(60)
    except KeyboardInterrupt:
        print("Monitoring stopped.")


if __name__ == "__main__":
    main()
