# HOG vs. SIFT on Blood-Cell Images

## Project Overview
This project presents an empirical evaluation and performance comparison between two classic Computer Vision feature extraction methods:
* **HOG (Histogram of Oriented Gradients)**
* **SIFT (Scale-Invariant Feature Transform)**

The evaluation is conducted on a subset of the **BloodMNIST** dataset to classify microscopic blood cell images into four target cellular categories using a **Linear Support Vector Machine (LinearSVC)**.

---

## Dataset & Experimental Parameters

### Dataset Details
* **Dataset Name:** `BloodMNIST` (via MedMNIST library)
* **Image Size:** Resized to **128×128** pixels (Grayscale).
* **Classes Evaluated (4 Selected Classes):**
  1. `1`: **eosinophil**
  2. `5`: **monocyte**
  3. `6`: **neutrophil**
  4. `7`: **platelet**

### Dataset Split Sizes
* **Training Set:** 400 images (100 images per class)
* **Validation Set:** 120 images (30 images per class)
* **Test Set:** 120 images (30 images per class)

### Key Configurations & Seeds
* **Registration Number:** `006`
* **Random Seed:** `6`
* **HOG Configuration:**
  * Cell Size: `8×8`
  * Orientations: `9`
  * Cells per Block: `(2, 2)`
* **SIFT Configuration:**
  * Grid Size: `6×6` (Dense sampling of 36 keypoints)
  * Keypoint Size: `32`
* **Classifier:** `LinearSVC` with L2 penalty, evaluated across hyperparameter values $C \in \{0.1, 1, 10\}$.

---

## Prerequisites & Dependencies

To execute the code in this repository, ensure Python 3.x is installed along with the required libraries:

```bash
pip install medmnist opencv-python-headless scikit-image scikit-learn seaborn pandas numpy matplotlib
```

---

## Code Workflow & Structure

### 1. Environment Setup & Configuration
Imports core scientific and computer vision packages (`cv2`, `skimage`, `sklearn`, `medmnist`), fixes seed values (`SEED = 6`) for full reproducibility, and defines constant parameters for dataset resizing and feature extraction configurations.

### 2. Dataset Preprocessing (`Q1. Data setup`)
* Downloads `bloodmnist` data using `medmnist` API.
* Subsamples exact per-class balances (100 train / 30 val / 30 test per class) via `choose_subset()`.
* Converts RGB or single-channel inputs to standard 2D grayscale matrices and resizes them to $128 \times 128$.

### 3. Feature Extraction Algorithms
* **HOG Feature Extraction:** Calculates gradient orientations and magnitudes over $8\times 8$ pixel cells and standardizes block normalization across $(2,2)$ cell blocks.
* **SIFT Feature Extraction:** Implements a dense grid sampling mechanism generating $6 \times 6 = 36$ fixed spatial keypoints per image and extracts 128-dimensional descriptor vectors per keypoint.

### 4. Classification Pipeline & Hyperparameter Tuning
* Constructs Scikit-Learn pipelines combining `StandardScaler` with `LinearSVC`.
* Sweeps regularization parameter $C \in [0.1, 1, 10]$ on the validation set for both feature pipelines.
* Evaluates trained models using Top-1 Accuracy, Macro F1-score, Classification Reports, and Confusion Matrices.

---

### Q5(a). Why does HOG use block normalisation?

HOG block normalisation reduces the effect of overall brightness and contrast changes by scaling gradient histograms within a local block. This makes the feature representation more stable under illumination changes.

### Q5(b). Why is it wrong to choose C using the test set?

Choosing C on the test set leaks information from the final evaluation data into model selection. This makes the reported test performance optimistically biased and no longer an independent estimate of generalisation; C should be selected using validation data only.

### Q5(c). Which two classes were confused most, and why might grayscale make this harder?

Use the class-pair outputs above and the validation confusion matrices to state which pair was confused most by your model. Grayscale removes color information that could help distinguish blood-cell types, so classes with similar shapes and textures may be harder to separate.

