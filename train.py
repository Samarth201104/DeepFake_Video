import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight

# =========================
# GPU setup
# =========================
print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print("⚠ Could not set memory growth:", e)

# Enable mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed precision enabled")

# =========================
# Paths
# =========================
base_dir = "preprocessed_split"
train_dir = os.path.join(base_dir, "train")  # contains real/ and fake/
val_dir   = os.path.join(base_dir, "val")    # contains real/ and fake/
test_dir  = os.path.join(base_dir, "test")   # contains real/ and fake/

# =========================
# Parameters
# =========================
img_size = (256, 256)
batch_size = 16  # reduce if GPU memory issues
epochs = 50
lr = 1e-4

# =========================
# Data augmentation
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

# =========================
# Generators
# =========================
# IMPORTANT: classes should match folder names directly under train_dir/val_dir
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    classes=['real', 'fake']
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    classes=['real', 'fake']
)

# =========================
# Compute class weights
# =========================
if len(train_generator.classes) > 0:
    class_weights_values = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights_values))
    print("Class Weights:", class_weights)
else:
    raise ValueError("No training images found. Check your folder structure!")

# =========================
# Build Xception model
# =========================
base_model = Xception(weights="imagenet", include_top=False, input_shape=(256, 256, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid", dtype='float32')(x)  # force float32 for output

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all but last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=lr),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =========================
# Callbacks
# =========================
checkpoint = ModelCheckpoint(
    "xception_best.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    verbose=1
)

# =========================
# Train
# =========================
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[checkpoint, earlystop, reduce_lr],
    verbose=1
)

print("\n✅ Training complete. Best model saved as xception_best.h5")
