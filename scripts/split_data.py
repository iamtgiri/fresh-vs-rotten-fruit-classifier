import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# === Configuration ===
SEED = 42
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

SOURCE_DIR = Path("data/dataset/Train")  # Path to the extracted training dataset
DEST_DIR = Path("data/split_dataset")    # Output directory for splits
TRAIN_DIR = DEST_DIR / "Train"
VAL_DIR = DEST_DIR / "Val"
TEST_DIR = DEST_DIR / "Test"

random.seed(SEED)


def prepare_split_dirs():
    if DEST_DIR.exists():
        print("Removing existing split directory...")
        shutil.rmtree(DEST_DIR)

    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        split_dir.mkdir(parents=True, exist_ok=True)


def split_dataset():
    print("Splitting dataset...")
    for class_dir in tqdm(list(SOURCE_DIR.iterdir()), desc="Processing classes"):
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*"))
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * SPLIT_RATIOS["train"])
        n_val = int(n_total * SPLIT_RATIOS["val"])
        n_test = n_total - n_train - n_val

        splits = {
            TRAIN_DIR / class_dir.name: images[:n_train],
            VAL_DIR / class_dir.name: images[n_train:n_train + n_val],
            TEST_DIR / class_dir.name: images[n_train + n_val:]
        }

        for dest_class_dir, files in splits.items():
            dest_class_dir.mkdir(parents=True, exist_ok=True)
            for file in files:
                shutil.copy(file, dest_class_dir)


if __name__ == "__main__":
    prepare_split_dirs()
    split_dataset()
    print("✅ Dataset split into train, val, and test sets.")
