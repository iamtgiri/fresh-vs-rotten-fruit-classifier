import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import mixed_precision
from pathlib import Path
import json

from utils.dataset_utils import load_dataset, get_class_names

# === Mixed Precision (optional) ===
mixed_precision.set_global_policy('mixed_float16')

# === Config ===
BATCH_SIZE = 8
EPOCHS = 30
SEED = 42

BASE_DIR = Path("data/split_dataset")
TRAIN_DIR = BASE_DIR / "Train"
VAL_DIR = BASE_DIR / "Val"
MODEL_PATH = Path("models/best_mobilenetv2.keras")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Reproducibility ===
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

# === Load Dataset ===
train_ds = load_dataset(TRAIN_DIR, seed=SEED, batch_size=BATCH_SIZE)
val_ds = load_dataset(VAL_DIR, seed=SEED, batch_size=BATCH_SIZE)
class_names = get_class_names(TRAIN_DIR)
num_classes = len(class_names)

# === Build MobileNetV2 Model ===
base_model = MobileNetV2(
    include_top=False,
    input_shape=(224, 224, 3),
    weights="imagenet",
    pooling="avg"
)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=True)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# === Callbacks ===
callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
]

# === Train ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# === Save Artifacts ===
(OUTPUT_DIR / "mobilenetv2_training_history.json").write_text(json.dumps(history.history))
(OUTPUT_DIR / "class_names.json").write_text(json.dumps(class_names))

print("\n✅ MobileNetV2 training complete. Best model saved to:", MODEL_PATH)
