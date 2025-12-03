# import tensorflow as tf
# print("TensorFlow version:", tf.__version__)
# print("Built with CUDA:", tf.test.is_built_with_cuda())
# print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
# print("GPU Details:", tf.config.list_physical_devices('GPU'))


import os
import time

# Paths
# real_dir = 'preprocessed/real'
fake_dir = 'preprocessed/fake'

# Total number of videos you planned to preprocess
# total_real_videos = len(os.listdir('dataset/real'))
total_fake_videos = len(os.listdir('dataset/fake'))  # or planned subset

def get_video_progress(output_dir):
    files = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
    # Keep everything except the last 2 underscore-separated parts
    video_names = set("_".join(f.split("_")[:-2]) for f in files)
    total_frames = len(files)
    return len(video_names), total_frames


try:
    while True:
        # real_videos_done, real_frames = get_video_progress(real_dir)
        fake_videos_done, fake_frames = get_video_progress(fake_dir)

        # print(f"📊 Real videos: {real_videos_done}/{total_real_videos} | Frames: {real_frames}")
        print(f"📊 Fake videos: {fake_videos_done}/{total_fake_videos} | Frames: {fake_frames}")
        print("-" * 50)

        time.sleep(60)  # update every 60 seconds

except KeyboardInterrupt:
    print("Monitoring stopped.")
