# Image Segmentation Challenge

Welcome to the image segmentation challenge! This project provides a baseline pipeline using scribble-based supervision for binary image segmentation. You will work with a dataset of natural images, sparse scribbles indicating foreground/background, and optionally ground truth labels for evaluation.

---

## Project Structure
```text
.
├── challenge.py           # Main script for training and testing pipeline
├── util.py                # Utility functions (dataset I/O, KNN model, visualization)
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   ├── scribbles/
│   │   ├── ground_truth/
│   │   └── predictions/   # Created by the script
│   └── test1/
│       ├── images/
│       ├── scribbles/
│       └── predictions/   # Created by the script
```

---

## Requirements

Install required Python packages via pip:

pip install numpy pillow matplotlib scikit-learn

---

## Running the Baseline

Execute the following script:

python challenge.py

This script will:

1. Load the training dataset.
2. Segment images using a K-Nearest Neighbors (KNN) classifier (k=3) based on scribble labels.
3. Store the predicted segmentation masks.
4. Visualize one randomly selected result.
5. Repeat the process on the test dataset (without ground truth).

---

## Contents

### challenge.py

- Loads training and test datasets.
- Performs segmentation using a baseline KNN model.
- Stores predicted masks to predictions/.
- Visualizes one result for the training set.

### util.py

A utility module containing:

- **Dataset Handling**
  - load_dataset(): Load images, scribbles, ground truths.
  - store_predictions(): Save predicted segmentation masks with color palettes.
- **Model**
  - segment_with_knn(): A KNN-based baseline using scribble supervision.
- **Visualization**
  - visualize(): Display image + scribbles, ground truth, and predicted mask.
- **Evaluation**
  - evaluate_binary_miou(): Computes mean Intersection over Union (IoU).

---


## Scribble Supervision Format

- Scribble masks use:
  - 0: Background
  - 1: Foreground
  - 255: Unlabeled

Predictions and ground truths are binary masks using 0 and 1.

---

## Tasks for Students

- Improve the segmentation performance beyond the KNN baseline.
- Try different models or distance metrics.
- Use the provided visualization and evaluation tools to test your ideas.
- Optionally, extend to multi-class segmentation (requires additional modification).

---

## Questions?

Please contact the course staff or teaching assistants for help with setup, debugging, or project expectations.

Happy coding!
