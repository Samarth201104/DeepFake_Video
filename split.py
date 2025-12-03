import os
import shutil
import random
import cv2

# Paths
input_base = "preprocessed"   # already preprocessed faces
output_base = "preprocessed_split"
os.makedirs(output_base, exist_ok=True)

# Split ratios
SPLITS = {'train': 0.7, 'val': 0.15, 'test': 0.15}

# Quality thresholds
MIN_SIZE = 30   # ignore images smaller than 60x60
BLUR_THRESH = 20  # lower = blurrier (adjust if needed)

def is_good_image(img_path):
    """Check if image has good quality (not too small, not blurry)."""
    img = cv2.imread(img_path)
    if img is None:
        return False

    h, w = img.shape[:2]
    if w < MIN_SIZE or h < MIN_SIZE:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < BLUR_THRESH:
        return False

    return True

def split_images(label):
    input_dir = os.path.join(input_base, label)
    images = [img for img in os.listdir(input_dir) if is_good_image(os.path.join(input_dir, img))]
    random.shuffle(images)

    total = len(images)
    train_split = int(SPLITS['train'] * total)
    val_split = int(SPLITS['val'] * total)

    split_data = {
        'train': images[:train_split],
        'val': images[train_split:train_split+val_split],
        'test': images[train_split+val_split:]
    }

    for split, files in split_data.items():
        split_dir = os.path.join(output_base, label, split)
        os.makedirs(split_dir, exist_ok=True)
        for img in files:
            src = os.path.join(input_dir, img)
            dst = os.path.join(split_dir, img)
            shutil.copy(src, dst)

    print(f"✅ {label}: {total} GOOD images split into "
          f"{len(split_data['train'])} train / {len(split_data['val'])} val / {len(split_data['test'])} test")

if __name__ == "__main__":
    for label in ["real", "fake"]:
        split_images(label)
    print("\n🎯 Splitting complete! Data ready in 'preprocessed_split/{real,fake}/{train,val,test}'")
