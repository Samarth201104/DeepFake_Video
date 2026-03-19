
import os
import cv2
import random
from retinaface import RetinaFace
from PIL import Image

# Input and output paths
fake_dir = 'dataset/fake'
output_base = 'preprocessed/fake'
os.makedirs(output_base, exist_ok=True)

# Parameters
image_size = (256, 256)
frame_skip = 15   # process every 15th frame
target_frames = 80066   # match real frames

def extract_faces(video_path, output_dir):
    """Extract faces from a video using RetinaFace (GPU)."""
    cap = cv2.VideoCapture(video_path)
    frame_count, saved_count = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            try:
                detections = RetinaFace.detect_faces(frame)
            except Exception:
                detections = None

            if isinstance(detections, dict):
                for i, face in enumerate(detections.values()):
                    x1, y1, x2, y2 = face['facial_area']
                    x1, y1 = max(0, x1), max(0, y1)
                    cropped_face = frame[y1:y2, x1:x2]

                    if cropped_face.size == 0:
                        continue

                    face_img = Image.fromarray(
                        cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                    ).resize(image_size)
                    save_path = os.path.join(
                        output_dir, f"{os.path.basename(video_path)}_{frame_count}_{i}.jpg"
                    )
                    face_img.save(save_path)
                    saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count

def preprocess_fake():
    # Count already saved faces
    already_saved = len([f for f in os.listdir(output_base) if f.endswith(".jpg")])
    print(f"🔄 Resuming... already have {already_saved} fake faces extracted")

    total_faces = already_saved
    total_videos = 0

    # Get fake video list and shuffle
    fake_videos = os.listdir(fake_dir)
    random.shuffle(fake_videos)

    # Skip videos that seem to be already processed (based on filenames)
    processed_prefixes = {f.split("_")[0] for f in os.listdir(output_base) if f.endswith(".jpg")}

    for video_name in fake_videos:
        if total_faces >= target_frames:
            break  # stop once we reach ~80k frames

        if os.path.splitext(video_name)[0] in processed_prefixes:
            continue  # skip this video, already processed before

        video_path = os.path.join(fake_dir, video_name)
        faces_saved = extract_faces(video_path, output_base)
        total_videos += 1
        total_faces += faces_saved
        print(f"✔ [FAKE] {video_name} → {faces_saved} faces saved (Total: {total_faces})")

    print("\n========== SUMMARY ==========")
    print(f"✅ New Fake Videos Processed: {total_videos}")
    print(f"✅ Total Fake Faces Extracted: {total_faces} (Target: {target_frames})")
    print("=============================\n")

if __name__ == "__main__":
    preprocess_fake()
