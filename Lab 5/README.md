# LAB 05: HOG-Based Industrial Defect Detection and Classification

This repository contains the notebook and documentation for **Lab 05: HOG-Based Industrial Defect Detection and Classification**. The project implements a complete computer vision and machine learning pipeline for detecting and classifying defects in industrial steel surface images using Histogram of Oriented Gradients (HOG) features along with Support Vector Machine (SVM) and Random Forest classifiers.

---

## 📌 Pipeline Overview

$$\text{Image} \longrightarrow \text{Preprocessing} \longrightarrow \text{HOG Feature Extraction} \longrightarrow \text{ML Classifier} \longrightarrow \text{Defective / Non-Defective / Defect Category Decision}$$

1. **Dataset Loading & Inspection:** Download and parse image assets and folder structures.
2. **Preprocessing:** Resize images to a uniform resolution ($128 \times 128$) and convert them to single-channel grayscale.
3. **Feature Extraction:** Compute HOG descriptors across varying cell sizes and orientation bins.
4. **Model Training & Comparison:** Evaluate SVM and Random Forest classifiers on the extracted feature vectors.
5. **Evaluation Metrics:** Generate confusion matrices, classification reports, accuracy, precision, recall, and F1-scores.
6. **Hyperparameter & Sensitivity Experiments:** Analyze performance across different HOG configurations and test robustness against image degradations (brightness shift, Gaussian noise, rotation, and blur).
7. **Industrial Quality-Control Decision Module:** Automatically flag items for rejection if a defect class is predicted.

---

## 🚀 Key Features & Activities

1. **Dataset Pipeline:** Integrated with `kagglehub` for automated dataset retrieval (`sovitrath/neu-steel-surface-defect-detect-trainvalid-split`).
2. **Preprocessing:** Batch resizing to $128 \times 128$ and grayscale conversion via OpenCV/scikit-image.
3. **HOG Visualizations:** Extraction and visualization of HOG gradient representations.
4. **Multi-Class Defect Classification:**
   - **SVM Classifier:** Linear/RBF Support Vector Machine training with scaled HOG features.
   - **Random Forest Classifier:** Ensemble learning evaluation.
5. **Comparative Analysis:** Detailed performance comparisons via confusion matrices, precision, recall, and F1-score.
6. **HOG Parameter Tuning Experiments:**
   - **Cell Sizes:** $4 \times 4$, $8 \times 8$, and $16 \times 16$.
   - **Orientations:** 6, 9, and 12 bins.
7. **Robustness Testing:** Evaluation under simulated industrial noise (brightness changes, Gaussian noise, rotational variance, and Gaussian blur).
8. **Quality Control Decision Engine:** Maps predictions to industrial actions (**DEFECTIVE $\rightarrow$ REJECT** / **NON-DEFECTIVE $\rightarrow$ ACCEPT**).

---

## 📊 Dataset Note

The default dataset used in this lab is the **NEU Steel Surface Defect Dataset** (NEU-DET), which contains six distinct defect categories:
* `crazing`
* `inclusion`
* `patches`
* `pitted_surface`
* `rolled-in_scale`
* `scratches`

> **Note:** The standard split of this dataset contains defect classes. The decision module interprets any predicted defect class as **DEFECTIVE $\rightarrow$ REJECT**. If a dataset variant containing a normal/non-defective class (e.g., `normal`, `good`, `ok`) is loaded, it is automatically assigned as Class `0` (**NON-DEFECTIVE $\rightarrow$ ACCEPT**).

---

## 🛠️ Requirements & Dependencies

The notebook requires Python 3.8+ and the following packages:

```bash
pip install opencv-python numpy pandas matplotlib seaborn scikit-image scikit-learn kagglehub