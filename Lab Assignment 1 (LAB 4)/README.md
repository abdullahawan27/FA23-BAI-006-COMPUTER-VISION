# Skin Cancer Image Processing, Edge Detection & Analysis (HAM10000)

This repository contains a comprehensive Jupyter Notebook (`lab assignment.ipynb`) covering dataset handling, spatial filtering, edge detection analysis, feature extraction, and machine learning classification on dermoscopic skin lesion images from the **HAM10000** ("Human Against Skin Cancer") dataset.

---

## 📌 Project Overview

This project implements classical computer vision techniques and modern classification workflows for automated skin lesion analysis. The lab is structured into 6 primary tasks:

1. **Dataset Download & Class Sampling:** Programmatic dataset retrieval and balanced diagnostic class sampling.
2. **Preprocessing & Spatial Filtering:** Color space transformations and Gaussian noise reduction.
3. **Edge Detection & Threshold Analysis:** Comparative evaluation of Sobel, Prewitt, Laplacian, LoG, and multi-threshold Canny edge operators.
4. **Lesion Segmentation & Feature Extraction:** ROI isolation and feature vector generation (color, texture, and shape descriptors).
5. **Classical Machine Learning:** Benchmarking SVM, Random Forest, and KNN classifiers across processed feature representations.
6. **Deep Learning & Model Comparison:** Training a Convolutional Neural Network (CNN) and comparing its diagnostic performance against classical models.

---

## 📊 Dataset Information

* **Dataset:** [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
* **Total Images:** 10,015 dermatoscopic images
* **Target Classes Selected:**
  | Diagnosis Code (`dx`) | Disease Name | Category |
  | :--- | :--- | :--- |
  | `mel` | Melanoma | Malignant |
  | `nv` | Melanocytic Nevi | Benign |
  | `bkl` | Benign Keratosis-like Lesions | Benign |
  | `bcc` | Basal Cell Carcinoma | Malignant |
  | `akiec` | Actinic Keratoses / Intraepithelial Carcinoma | Pre-malignant / Malignant |

---

## 🛠️ Task Summaries

### Task 1: Dataset Setup & Class Sampling
* Programmatically downloads the dataset using `kagglehub`.
* Scans directory paths for `HAM10000_metadata.csv` and matching `*.jpg` files.
* Constructs a lookup table mapping `image_id` to absolute file paths.
* Filters metadata and extracts representative sample images for each of the 5 target classes (`mel`, `nv`, `bkl`, `bcc`, `akiec`).

### Task 2: Image Preprocessing & Spatial Filtering
* Converts default OpenCV BGR images to RGB for visualization.
* Transforms color images into Grayscale (`cv2.COLOR_BGR2GRAY`).
* Applies a $5 \times 5$ **Gaussian Smoothing Filter** (`cv2.GaussianBlur`) to eliminate high-frequency noise (e.g., skin pores, fine texture) while preserving macroscopic lesion boundaries.
* Displays 3-panel comparative plots (Original RGB vs. Grayscale vs. Gaussian Filtered).

### Task 3: Edge Detection Analysis
Applies gradient and derivative edge detection operators:
* **First-Order Operators:** Sobel ($x$, $y$, and combined magnitude) and Prewitt filters.
* **Second-Order Operators:** Laplacian filter and Laplacian of Gaussian (LoG).
* **Multi-Stage Canny Detection:** Benchmarking three distinct hysteresis threshold pairs ($T_{\text{low}}, T_{\text{high}}$) to determine optimal lesion boundary isolation.

### Task 4: Lesion Segmentation & Feature Extraction
* Isolates lesion regions of interest (ROI) via thresholding and Otsu's binarization.
* Extracts multi-dimensional feature sets:
  * **Color Metrics:** Channel-wise mean, standard deviation, and HSV/RGB color histograms.
  * **Texture Metrics:** Gray-Level Co-occurrence Matrix (GLCM) features (contrast, dissimilarity, homogeneity, energy, correlation).
  * **Shape Descriptors:** Area, perimeter, circularity, and eccentricity.

### Task 5: Classical Machine Learning Classification
* Splits feature matrices into train/test sets ($80/20$ ratio) with class stratification.
* Trains and evaluates classifiers:
  * **Support Vector Machine (SVM)** (RBF kernel)
  * **Random Forest Classifier**
  * **K-Nearest Neighbors (KNN)**
* Reports Accuracy, Precision, Recall, and F1-score across raw, filtered, and edge-extracted features.

### Task 6: Deep Learning (CNN) & Comparative Analysis
* Implements and trains a Convolutional Neural Network (CNN) architecture.
* Incorporates data augmentation (rotation, flipping, zooming) to handle class imbalance.
* Plots training/validation loss and accuracy curves.
* Generates confusion matrices and compares CNN accuracy against Task 5 classical machine learning pipelines.

---

## ❓ Edge Detection Lab Analysis & Discussion
## ❓ Questions to Answer

### 1. Why is Gaussian filtering applied before Canny detection?
Canny edge detection relies on first-order intensity gradients. Differential operators amplify high-frequency spatial variations, making raw images susceptible to false edge detections from skin pores, fine hairs, and surface texture. A **Gaussian filter** smooths out high-frequency noise while preserving true lesion boundaries.

### 2. How did the three Canny threshold settings affect the result?
* **Low Thresholds ($30/90$):** Retained weak gradients, resulting in over-segmentation with significant background noise, skin texture, and hair fragments.
* **High Thresholds ($100/200$):** Captured only extreme gradient changes, causing under-segmentation with broken, discontinuous lesion contours.
* **Moderate Thresholds ($50/150$):** Balanced noise suppression and boundary continuity, yielding a clear outer lesion contour.

### 3. Which threshold produced the best lesion boundary?
The **intermediate threshold pair ($T_{\text{low}}=50, T_{\text{high}}=150$)** produced the cleanest boundary, successfully outlining the lesion margin without picking up spurious skin noise.

### 4. Why are edges useful for detecting skin lesions?
Edges define the perimeter separating healthy skin from pathological tissue. They are essential for isolating the Region of Interest (ROI) and quantifying clinical **ABCD rule** parameters (Asymmetry, Border irregularity, Color variation, Diameter), such as boundary roughness, compactness, and surface area.

### 5. What problems were observed in detecting the lesion boundary?
* **Hair Interference:** Dark hairs crossing the lesion create sharp local contrast gradients that trigger false edges.
* **Fuzzy Margins:** Low-contrast lesions (e.g., early-stage nevi) fade gradually into healthy skin, leading to boundary gaps.
* **Internal Pigmentation:** Variegated pigment spots inside the lesion body generate unwanted internal edges.
* **Vignetting:** Light falloff around image corners can introduce peripheral border artifacts.

### 6. How could the method be improved?
* **Hair Removal Preprocessing:** Implement Black-Hat morphological filtering or the DullRazor algorithm prior to smoothing.
* **Alternative Color Spaces:** Run edge detection on individual channels in $L^*a^*b^*$ ($a^*/b^*$ channels) or HSV ($S$ channel), where skin-to-lesion contrast is higher than in grayscale.
* **Adaptive Thresholding:** Dynamic parameter selection using image median statistics ($T_{\text{low}} = 0.66 \times \text{median}, T_{\text{high}} = 1.33 \times \text{median}$) or Otsu's thresholding.
* **Morphological Closing:** Post-process edge maps with dilation and erosion to bridge boundary gaps and eliminate isolated noise points.

---

## 📁 Repository Structure

```text
├── lab assignment.ipynb     # Main Jupyter notebook with all 6 tasks
├── README.md               # Project documentation and lab answers
├── requirements.txt        # Python dependency list
└── data/                   # (Auto-generated) Cache directory for HAM10000 dataset