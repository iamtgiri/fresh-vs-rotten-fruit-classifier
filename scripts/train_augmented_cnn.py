import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import json

from utils.dataset_utils import load_dataset

# === Config ===
BATCH_SIZE = 64
EPOCHS = 20
SEED = 42

BASE_DIR = Path("data/split_dataset")
TRAIN_DIR = BASE_DIR / "Train"
VAL_DIR = BASE_DIR / "Val"
BASIC_MODEL_PATH = Path("models/best_basic_model.keras")
AUG_MODEL_PATH = Path("models/best_aug_model.keras")

# === Reproducibility ===
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# === Load Datasets with Augmentation ===
train_ds_aug = load_dataset(TRAIN_DIR, seed=SEED, batch_size=BATCH_SIZE, augment=True)
val_ds = load_dataset(VAL_DIR, seed=SEED, batch_size=BATCH_SIZE)

# === Load base model ===
model = load_model(BASIC_MODEL_PATH)

# === Callbacks ===
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=AUG_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
]

# === Train ===
history = model.fit(
    train_ds_aug,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\n✅ Augmented training complete. Best model saved to:", AUG_MODEL_PATH)
