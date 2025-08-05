import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path

# === Constants ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

normalization_layer = layers.Rescaling(1./255)

# === Dataset Loaders ===
def load_dataset(directory, seed=42, batch_size=BATCH_SIZE, augment=False, shuffle=True):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        seed=seed,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=shuffle
    )

    if augment:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.05),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
        ])
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    ds = ds.map(lambda x, y: (normalization_layer(x), y))
    return ds.cache().prefetch(AUTOTUNE)


def get_class_names(directory):
    temp_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        seed=42,
        image_size=IMG_SIZE,
        batch_size=1
    )
    return temp_ds.class_names
