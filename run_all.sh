#!/bin/bash

set -e

echo "============================="
echo " Fresh vs Rotten Classifier"
echo "============================="

# Step 1: Split dataset
echo "\n[1/5] Splitting dataset..."
python3 scripts/split_data.py

# Step 2: Train basic CNN
echo "\n[2/5] Training basic CNN model..."
python3 scripts/train_cnn.py

# Step 3: Train augmented CNN
echo "\n[3/5] Training augmented CNN model..."
python3 scripts/train_augmented_cnn.py

# Step 4: Train MobileNetV2 model
echo "\n[4/5] Training MobileNetV2 model..."
python3 scripts/train_mobilenetv2.py

# Step 5: Evaluate best model
echo "\n[5/5] Evaluating MobileNetV2 model..."
python3 scripts/evaluate.py

echo "\n✅ All tasks completed successfully!"