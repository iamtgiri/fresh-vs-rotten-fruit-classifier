import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from pathlib import Path

# === Config ===
IMG_SIZE = (224, 224)
MODEL_PATH = Path("models/best_mobilenetv2.keras")  # change if needed
CLASS_NAMES_PATH = Path("models/class_names.json")
UNSEEN_DIR = Path("data/dataset/Test")  # manually extracted test folder from Kaggle
NUM_SAMPLES = 10

# === Load model and class names ===
model = load_model(MODEL_PATH)
class_names = json.loads(CLASS_NAMES_PATH.read_text())

# === Collect image paths ===
image_paths = []
for root, _, files in os.walk(UNSEEN_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, file))

# === Pick random sample ===
sample_paths = random.sample(image_paths, NUM_SAMPLES)

# === Predict and Plot ===
plt.figure(figsize=(20, 10))

for i, img_path in enumerate(sample_paths):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0  # normalize
    img_batch = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_batch, verbose=0)
    pred_class = class_names[np.argmax(preds)]

    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(f"Predicted: {pred_class}", fontsize=10)
    plt.axis("off")

plt.tight_layout()
plt.show()

print(f"\n✅ Displayed predictions for {NUM_SAMPLES} random images from:", UNSEEN_DIR)
