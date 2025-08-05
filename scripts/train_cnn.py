import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout,
                                     BatchNormalization, Input, GlobalAveragePooling2D)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import json

from utils.dataset_utils import load_dataset, get_class_names

# === Config ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 20
SEED = 42

BASE_DIR = Path("data/split_dataset")
TRAIN_DIR = BASE_DIR / "Train"
VAL_DIR = BASE_DIR / "Val"
MODEL_PATH = Path("models/best_basic_model.keras")

# === Reproducibility ===
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# === Load Datasets ===
train_ds = load_dataset(TRAIN_DIR, seed=SEED, batch_size=BATCH_SIZE)
val_ds = load_dataset(VAL_DIR, seed=SEED, batch_size=BATCH_SIZE)
class_names = get_class_names(TRAIN_DIR)

# === Build CNN Model ===
def build_cnn(input_shape=(224, 224, 3), num_classes=len(class_names)):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# === Compile & Train ===
model = build_cnn()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# === Save class names ===
(Path("models") / "class_names.json").write_text(json.dumps(class_names))

print("\n✅ Training complete. Best model saved to:", MODEL_PATH)
