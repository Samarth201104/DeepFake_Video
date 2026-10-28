"""Quality-filter and split face crops into train, validation, and test sets."""
import argparse
import random
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}
MIN_SIZE = 30
BLUR_THRESH = 20


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "processed" / "faces")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "splits")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible image ordering.")
    return parser.parse_args()


def is_good_image(img_path):
    """Accept images at least 30px in both dimensions with Laplacian variance >=20."""
    image = cv2.imread(str(img_path))
    if image is None:
        return False
    height, width = image.shape[:2]
    if width < MIN_SIZE or height < MIN_SIZE:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= BLUR_THRESH


def split_images(label, input_base, output_base, rng):
    input_dir = input_base / label
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input label directory not found: {input_dir}")
    images = [path.name for path in input_dir.iterdir() if path.is_file() and is_good_image(path)]
    rng.shuffle(images)

    total = len(images)
    train_end = int(SPLITS["train"] * total)
    val_end = train_end + int(SPLITS["val"] * total)
    split_data = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }
    for split, filenames in split_data.items():
        split_dir = output_base / split / label
        split_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            shutil.copy2(input_dir / filename, split_dir / filename)
    print(
        f"{label}: {total} quality images split into "
        f"{len(split_data['train'])} train / {len(split_data['val'])} val / {len(split_data['test'])} test"
    )


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    for label in ("real", "fake"):
        split_images(label, args.input_dir, args.output_dir, rng)
    print(f"Dataset splits written to {args.output_dir}")


if __name__ == "__main__":
    main()
