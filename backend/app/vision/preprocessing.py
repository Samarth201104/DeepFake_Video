"""Shared image resizing and normalization for the 256px classifier input."""
import numpy as np
from PIL import Image

IMAGE_SIZE = (256, 256)


def resize_face(rgb_face, target_size=IMAGE_SIZE):
    return np.asarray(Image.fromarray(rgb_face).resize(target_size))


def normalize_image(image):
    """Match the training/evaluation rescale=1/255 preprocessing."""
    return np.asarray(image, dtype=np.float32) / 255.0
