# Lab 07: Improving HOG-Based Medical Image Classification Using a Genetic Algorithm

## 1. Introduction
This project implements a medical image classification system using **Histogram of Oriented Gradients (HOG)** feature extraction and a **Support Vector Machine (SVM)** classifier. A **Genetic Algorithm (GA)** is integrated into the traditional HOG pipeline to perform feature selection, evaluating whether selecting a subset of features can improve classification performance while reducing dimensionality.

The project compares two distinct approaches:
1. **Baseline HOG + SVM**: Uses all extracted HOG features for classification.
2. **Proposed GA-HOG + SVM**: Uses a Genetic Algorithm to select an optimal subset of HOG features before training the SVM classifier.

Both approaches share the same training, validation, and test splits to ensure a fair and controlled comparison.

---

## 2. Objectives
- Download and preprocess the **Medical MNIST Dataset** from Kaggle.
- Preprocess medical images and split the data into stratified training, validation, and test sets.
- Implement a baseline HOG + SVM classification pipeline.
- Extract gradient-based features using scikit-image HOG extraction.
- Implement a Genetic Algorithm (GA) specifically designed for binary feature selection.
- Formulate a fitness function utilizing validation macro-F1 score penalized by feature count.
- Train and hyperparameter-tune SVM models on full vs. reduced feature sets.
- Evaluate and compare models using comprehensive classification metrics.
- Visualize fitness progression across GA generations and dimensionality reduction effects.
- Analyze trade-offs between computational overhead and performance gains.

---

## 3. Dataset & Preprocessing

### Dataset Information
* **Name**: Medical MNIST Dataset
* **Source**: Kaggle ([Dataset Link](https://www.kaggle.com/datasets/andrewmvd/medical-mnist))
* **Download Method**: Downloaded automatically via `kagglehub`. API token (`kaggle.json`) may be required depending on Kaggle authentication settings.

### Preprocessing Steps
1. **Image Loading**: Automatically identifies class folders and loads valid image formats.
2. **Grayscale Conversion**: Converts images to single-channel grayscale.
3. **Resizing**: Standardizes image dimensions to $128 \times 128$ pixels.
4. **Label Encoding**: Maps categorical directory labels to integer class labels.
5. **Stratified Splitting**:
   - **Training Set**: ~70%
   - **Validation Set**: ~15%
   - **Test Set**: ~15%

---

## 4. Technologies and Libraries
* **Python 3.x**
* **NumPy**: Vectorized numerical computations and array manipulations.
* **Pandas**: Structured data management and exporting metrics tables.
* **Matplotlib**: Visualizing sample images, GA convergence curves, and feature count comparisons.
* **Pillow (PIL)**: Basic image processing operations (loading, resizing, grayscale conversion).
* **scikit-image**: Feature extraction using Histogram of Oriented Gradients (`skimage.feature.hog`).
* **scikit-learn**: SVM classification (`SVC`), data splitting (`train_test_split`), and evaluation metrics.
* **kagglehub**: Direct programmatic downloading of Kaggle datasets.
* **IPython**: Execution environment display utilities.

---

## 5. HOG Feature Extraction
The HOG descriptor captures local shape and edge structure through gradient orientations. The configuration parameters applied consistently across both experiments include:

| Parameter | Value |
| :--- | :--- |
| **Image Resolution** | $128 \times 128$ |
| **Orientations** | 9 |
| **Pixels per Cell** | $8 \times 8$ |
| **Cells per Block** | $2 \times 2$ |
| **Block Normalization** | L2-Hys |
| **Feature Selection (Baseline)** | None (All features used) |
| **Feature Selection (Proposed)** | Genetic Algorithm Mask |

---

## 6. Baseline HOG + SVM
1. **Feature Vector**: Uses the complete standard HOG feature vector.
2. **Classifier**: Support Vector Machine with Radial Basis Function (RBF) kernel.
3. **Hyperparameter Selection**: Evaluates $C \in \{0.1, 1, 10\}$ on the validation set using **macro-F1 score**. The test set remains completely untouched during hyperparameter tuning.

---

## 7. Genetic Algorithm for Feature Selection

Each GA chromosome is represented as a **binary vector** where $1$ indicates a selected feature and $0$ indicates an omitted feature.

### GA Mechanics
1. **Population Initialization**: Random binary masks combined with specific seed chromosomes (e.g., full feature mask and sparse feature mask) to balance initial diversity and coverage.
2. **Fitness Function**:
   $$\text{Fitness} = \text{Validation Macro-F1} - \text{Penalty}$$
   $$\text{Penalty} = 0.05 \times \left( \frac{\text{Selected Features}}{\text{Total Features}} \right)$$
3. **Selection**: Tournament selection strategy.
4. **Crossover**: Single-point crossover combining parent chromosomes.
5. **Mutation**: Bit-flip mutation based on defined mutation probability.
6. **Elitism**: Retains the single best chromosome from each generation without mutation.
7. **Termination**: Runs for a set number of generations, returning the best mask found.

---

## 8. GA Experimental Configurations

To evaluate parameter sensitivity, two GA configurations are benchmarked against validation fitness:

| Parameter | GA-A (Smaller Population) | GA-B (Higher Mutation) |
| :--- | :--- | :--- |
| **Population Size** | 8 | 10 |
| **Generations** | 5 | 0.05 |
| **Mutation Probability** | 0.01 | 0.05 |

The configuration producing the highest validation fitness score is selected to train the final **GA-HOG + SVM** pipeline.

---

## 9. Performance Evaluation Metrics

Models are evaluated on the held-out test set across the following comprehensive criteria:

* **Accuracy**: Overall fraction of correct predictions.
* **Macro Precision / Recall / F1**: Unweighted per-class metric averages.
* **Weighted F1**: Class-frequency weighted F1 score.
* **Balanced Accuracy**: Average recall achieved per class.
* **Macro Specificity**: One-vs-Rest true negative rate averaged across classes.
* **Cohen's Kappa ($\kappa$)**: Inter-rater agreement corrected for random chance.
* **Matthews Correlation Coefficient (MCC)**: Quality metric taking into account true and false positives and negatives.
* **Computational Time**: Execution wall-clock time for search, training, and testing phases.

---

## 10. Results Summary

> **Note**: Specific values are computed dynamically upon running the notebook cells.

| Metric | Baseline HOG + SVM | Proposed GA-HOG + SVM |
| :--- | :--- | :--- |
| **Accuracy** | *Generated at runtime* | *Generated at runtime* |
| **Macro Precision** | *Generated at runtime* | *Generated at runtime* |
| **Macro Recall** | *Generated at runtime* | *Generated at runtime* |
| **Macro-F1** | *Generated at runtime* | *Generated at runtime* |
| **Weighted-F1** | *Generated at runtime* | *Generated at runtime* |
| **Balanced Accuracy** | *Generated at runtime* | *Generated at runtime* |
| **Macro Specificity** | *Generated at runtime* | *Generated at runtime* |
| **Cohen's Kappa** | *Generated at runtime* | *Generated at runtime* |
| **MCC** | *Generated at runtime* | *Generated at runtime* |
| **Test Prediction Time** | *Generated at runtime* | *Generated at runtime* |
| **Feature Count** | 100% of HOG features | Reduced GA Feature Mask |

---

## 11. Execution & Setup

### Prerequisites
Ensure Python 3.8+ is installed along with Jupyter Notebook or Google Colab.

### Installation
Run the following command to install required dependencies:
```bash
pip install numpy pandas matplotlib pillow scikit-image scikit-learn kagglehub
```

### Running the Project
1. Launch Jupyter Notebook or open Google Colab:
   ```bash
   jupyter notebook Lab_07_HOG_Genetic_Algorithm.ipynb
   ```
2. Execute cells sequentially from top to bottom.
3. Provide Kaggle API credentials (`kaggle.json`) if prompted during dataset retrieval.

---

## 12. Generated Artifacts
Execution automatically exports data and visualization logs:
* `lab07_test_results.csv`: Test set evaluation matrix comparing Baseline vs. Proposed.
* `lab07_feature_counts.csv`: Dimension comparison before and after GA mask application.
* `GA fitness history CSV files`: Detailed generation-by-generation tracking logs.

---

## 13. References
1. **Medical MNIST Dataset**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/andrewmvd/medical-mnist)
2. **scikit-image Documentation**: [HOG Feature Extraction](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog)
3. **scikit-learn Documentation**: [Support Vector Machines (SVM)](https://scikit-learn.org/stable/modules/svm.html)
4. **scikit-learn Metrics**: [Model Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
5. **KaggleHub Documentation**: [KaggleHub Repository](https://github.com/Kaggle/kagglehub)