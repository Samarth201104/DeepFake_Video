"""Train the repository's Xception binary face classifier."""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from backend.app.vision.preprocessing import normalize_image

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed" / "splits")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "models" / "xception_best.h5")
    return parser.parse_args()


def main():
    args = parse_args()
    print("TensorFlow version:", tf.__version__)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as error:
            print("Could not set memory growth:", error)

    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision enabled")

    image_size = (256, 256)
    batch_size = 16
    epochs = 50
    learning_rate = 1e-4
    train_datagen = ImageDataGenerator(
        preprocessing_function=normalize_image,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    validation_datagen = ImageDataGenerator(preprocessing_function=normalize_image)
    train_generator = train_datagen.flow_from_directory(
        args.data_dir / "train", target_size=image_size, batch_size=batch_size,
        class_mode="binary", classes=["real", "fake"],
    )
    validation_generator = validation_datagen.flow_from_directory(
        args.data_dir / "val", target_size=image_size, batch_size=batch_size,
        class_mode="binary", classes=["real", "fake"],
    )

    if len(train_generator.classes) == 0:
        raise ValueError(f"No training images found under {args.data_dir / 'train'}")
    values = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(train_generator.classes), y=train_generator.classes
    )
    class_weights = dict(enumerate(values))
    print("Class weights:", class_weights)

    base_model = Xception(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
    features = GlobalAveragePooling2D()(base_model.output)
    features = Dropout(0.5)(features)
    predictions = Dense(1, activation="sigmoid", dtype="float32")(features)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(str(args.checkpoint), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
    ]
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"Training complete. Best model saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
