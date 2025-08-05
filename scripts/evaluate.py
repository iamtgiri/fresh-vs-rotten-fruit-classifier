import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# === Config ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
MODEL_PATH = Path("models/best_mobilenetv2.keras")  # change to other models if needed
CLASS_NAMES_PATH = Path("models/class_names.json")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path("data/split_dataset")
TEST_DIR = BASE_DIR / "Test"

# === Load class names ===
class_names = json.loads(CLASS_NAMES_PATH.read_text())

# === Load Dataset ===
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)

# === Load Model ===
model = tf.keras.models.load_model(MODEL_PATH)

# === Evaluate ===
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\nTest Accuracy: {test_acc:.2%}")
print(f"Test Loss: {test_loss:.4f}")

# Save metrics to file
with open(OUTPUT_DIR / "mobilenetv2_test_metrics.txt", "w") as f:
    f.write(f"Test Accuracy: {test_acc:.4f}\nTest Loss: {test_loss:.4f}\n")

# === Predictions ===
y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# === Classification Report ===
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:\n", report)

with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
    f.write(report)

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
np.save(OUTPUT_DIR / "confusion_matrix.npy", cm)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.show()

# === Plot Sample Predictions ===
def denormalize(image):
    return np.clip(image * 255.0, 0, 255).astype("uint8")

test_images = np.concatenate([x.numpy() for x, _ in test_ds])

def plot_sample_predictions(images, y_true, y_pred, class_names, num_samples=12):
    indices = np.random.choice(len(images), num_samples, replace=False)
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        ax = plt.subplot(3, 4, i + 1)
        image = denormalize(images[idx])
        plt.imshow(image)
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        color = "green" if y_true[idx] == y_pred[idx] else "red"
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions_sample_grid.png")
    plt.show()

plot_sample_predictions(test_images, y_true, y_pred, class_names)
