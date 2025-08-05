# 🍎 Fresh vs Rotten Fruit Classifier

This project classifies fruits as **fresh** or **rotten** using Convolutional Neural Networks (CNNs) and Transfer Learning with **MobileNetV2**. It supports both basic and augmented training setups and exports various evaluation outputs for further analysis.

---

## 📦 Dataset

- **Source**: [Kaggle - Fresh and Stale Fruit Classification](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification/data)
- **How to Use**:
  1. Download the dataset manually from Kaggle.
  2. Extract it into the `data/` directory so that it looks like:

```

data/
└── dataset/
├── Train/
│   ├── freshapples/
│   ├── rottenapples/
│   └── ...
└── Test/
├── freshbanana/
├── rottenbanana/
└── ...

```

---

## ✅ Project Directory Structure

```

fresh-vs-rotten-fruit-classifier/
├── data/                    # Raw + processed dataset
│   ├── dataset/             # Extracted dataset from Kaggle
│   └── split\_dataset/       # Train/Val/Test split (generated)
│
├── notebooks/
│   └── EDA\_and\_Experiments.ipynb  # Optional exploratory analysis
│
├── scripts/
│   ├── split\_data.py              # Train/val/test splitter
│   ├── train\_cnn.py               # Custom CNN training
│   ├── train\_augmented\_cnn.py     # CNN + data augmentation
│   ├── train\_mobilenetv2.py       # Transfer learning (MobileNetV2)
│   ├── evaluate.py                # Evaluation & reports
│   └── predict\_random\_samples.py  # Predictions on unseen images
│
├── models/
│   └── \*.keras                    # Trained models
│
├── outputs/
│   ├── confusion\_matrix.png
│   ├── classification\_report.txt
│   ├── predictions\_sample\_grid.png
│   ├── mobilenetv2\_training\_history.json
│   ├── mobilenetv2\_test\_metrics.txt
│   └── \*.onnx, \*.tflite (optional exported models)
│
├── utils/
│   └── dataset\_utils.py           # Modular dataset loader
│
├── .gitignore
├── README.md
├── requirements.txt               # Dependencies
└── run\_all.sh                     # End-to-end execution

```

---

## 🛠️ How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Split the Dataset

```bash
python scripts/split_data.py
```

### 3. Train Custom CNN

```bash
python scripts/train_cnn.py
```

### 4. Fine-tune with Augmentation

```bash
python scripts/train_augmented_cnn.py
```

### 5. Train with MobileNetV2 (Transfer Learning)

```bash
python scripts/train_mobilenetv2.py
```

### 6. Evaluate the Trained Model

```bash
python scripts/evaluate.py
```

### 7. Make Predictions on Random Test Samples

```bash
python scripts/predict_random_samples.py
```

### 🔁 (Optional) Run Everything at Once

```bash
chmod +x run_all.sh
./run_all.sh
```

---

## 📊 Output Artifacts

Upon execution, the following outputs will be stored in the `outputs/` directory:

* **Training Metrics**

  * `mobilenetv2_training_history.json`
  * Accuracy/Loss curves
* **Evaluation Results**

  * `classification_report.txt`
  * `confusion_matrix.png`
  * `confusion_matrix.npy`
  * `mobilenetv2_test_metrics.txt`
* **Visualizations**

  * `predictions_sample_grid.png`
* **Model Exports**

  * `.keras` (Keras saved models)
  * `.onnx`, `.tflite` (optional exports)

---

## 💡 Notes

* All scripts are modular and reusable across model types.
* Dataset utilities (`utils/dataset_utils.py`) handle preprocessing, augmentation, and batching.
* Models are saved in `models/` directory and reused where applicable.

---

## 📌 Requirements

See [`requirements.txt`](./requirements.txt) for the full list of required packages.

---

## 📬 Contact

For queries or improvements, feel free to DM me on [LinkedIn](https://www.linkedin.com/in/iamtgiri/) or open an issue on this repository.

