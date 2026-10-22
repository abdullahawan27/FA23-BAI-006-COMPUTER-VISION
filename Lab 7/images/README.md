# Lab 07: Improving HOG-Based Medical Image Classification Using a Genetic Algorithm

## 1. Introduction

This project implements a medical image classification system using Histogram of Oriented Gradients (HOG) feature extraction and Support Vector Machine (SVM) classification. A Genetic Algorithm (GA) is integrated into the traditional HOG pipeline to select a subset of the extracted features and investigate whether feature selection can improve classification performance while reducing dimensionality.

The project compares two approaches:

1. **Baseline HOG + SVM:** Uses all extracted HOG features for classification.
2. **Proposed GA-HOG + SVM:** Uses a Genetic Algorithm to select a subset of HOG features before training the SVM classifier.

Both approaches use the same training, validation, and test splits to ensure a fair comparison.

## 2. Objectives

The main objectives of this lab are:

- Select and download a suitable medical image dataset from Kaggle.
- Preprocess medical images and divide the dataset into training, validation, and test sets.
- Implement a baseline HOG + SVM classification pipeline.
- Extract HOG features using the specified parameters.
- Implement a Genetic Algorithm for feature selection.
- Evaluate chromosomes using validation macro-F1 and a feature-selection penalty.
- Train an SVM classifier using the selected features.
- Compare the baseline and proposed models using multiple classification metrics.
- Visualize GA fitness across generations and compare feature dimensionality before and after selection.
- Analyze the strengths, limitations, and computational cost of GA-based feature selection.

## 3. Dataset

**Dataset name:** Medical MNIST Dataset

**Dataset source:** Kaggle

**Dataset link:** https://www.kaggle.com/datasets/andrewmvd/medical-mnist

The notebook downloads the dataset using the `kagglehub` Python library. If Kaggle authentication is required, a Kaggle API token (`kaggle.json`) may need to be configured.

The dataset contains medical image categories organized into class folders. The notebook automatically identifies the class folders, loads supported image formats, and assigns numeric labels to the classes.

To reduce memory usage and computational time, the notebook uses a configurable maximum number of images per class.

### Dataset preprocessing

The following preprocessing steps are performed:

1. Load images from the downloaded dataset.
2. Convert images to grayscale.
3. Resize images to 128 × 128 pixels.
4. Assign numeric labels to the image classes.
5. Create fixed training, validation, and test sets using stratified splitting.

The dataset is split into approximately:

- Training set: 70%
- Validation set: 15%
- Test set: 15%

The same split indices are used for both classification approaches.

## 4. Technologies and Libraries

The project uses Python and the following libraries:

- **NumPy:** Numerical operations and feature arrays.
- **Pandas:** Experimental results and comparison tables.
- **Matplotlib:** Visualization of images, GA fitness, and feature counts.
- **Pillow:** Image loading, grayscale conversion, and resizing.
- **scikit-image:** HOG feature extraction.
- **scikit-learn:** SVM classification, dataset splitting, and performance evaluation.
- **kagglehub:** Downloading the dataset from Kaggle.
- **IPython:** Displaying notebook outputs.

## 5. HOG Feature Extraction

Histogram of Oriented Gradients is a feature-extraction technique that represents image structure using the distribution of local gradient orientations.

In this project, HOG extracts gradient-based information from each medical image. These features are subsequently used by an SVM classifier.

### HOG parameters

| Parameter | Value |
|---|---|
| Image resolution | 128 × 128 |
| Orientations | 9 |
| Pixels per cell | 8 × 8 |
| Cells per block | 2 × 2 |
| Block normalization | L2-Hys |
| Feature selection | None in the baseline model |
| Feature selection in proposed model | Genetic Algorithm |

The same HOG configuration is used for both approaches.

## 6. Baseline HOG + SVM

The baseline approach extracts HOG features from the images and trains a Support Vector Machine classifier using the complete feature vector.

The SVM uses an RBF kernel to model nonlinear relationships between feature vectors and class labels.

The notebook evaluates the following values of the SVM regularization parameter:

- C = 0.1
- C = 1
- C = 10

The best parameter is selected using validation macro-F1. The test set is not used for hyperparameter selection.

The baseline model serves as the reference for evaluating the proposed GA-based feature-selection method.

## 7. Genetic Algorithm for Feature Selection

A Genetic Algorithm is an optimization technique inspired by natural selection and evolution.

In this project, each chromosome represents a binary feature-selection mask:

- `1` indicates that a HOG feature is selected.
- `0` indicates that a HOG feature is excluded.

The GA searches for a feature subset that balances classification performance and dimensionality reduction.

### GA components

**1. Population initialization**

An initial population of binary chromosomes is generated randomly. Additional initial chromosomes provide diversity by representing a compact feature subset and the complete feature set.

**2. Fitness evaluation**

Each chromosome is evaluated by training an SVM on its selected features and calculating macro-F1 on the validation set.

The fitness function is:

Fitness = Validation Macro-F1 − Feature Selection Penalty

The feature-selection penalty is proportional to the fraction of selected features:

Penalty = 0.05 × (Selected Features / Total Features)

This encourages the algorithm to find smaller feature subsets without ignoring classification performance.

**3. Selection**

Tournament selection chooses parent chromosomes based on their fitness values.

**4. Crossover**

One-point crossover combines portions of two parent chromosomes to generate offspring.

**5. Mutation**

Bit-flip mutation randomly changes selected and unselected feature indicators according to the mutation probability.

**6. Elitism**

The best chromosome from each generation is retained to preserve the best solution found so far.

**7. Termination**

The algorithm runs for a predefined number of generations and returns the best feature-selection mask found.

## 8. GA Experimental Configurations

The notebook evaluates two GA configurations to investigate how GA settings affect feature selection.

| Configuration | Population Size | Generations | Mutation Probability |
|---|---:|---:|---:|
| GA-A: Smaller population | 8 | 5 | 0.01 |
| GA-B: Higher mutation | 10 | 5 | 0.05 |

The configurations are compared using their validation-based fitness values.

The configuration with the highest fitness is selected for the final GA-HOG + SVM model.

## 9. GA-HOG + SVM Classification

After feature selection, the selected HOG features are used to train an SVM classifier.

The classifier uses the RBF kernel and the SVM regularization parameter selected during baseline validation.

The same training and validation/test partitions are maintained throughout the experiment.

The final model is evaluated on the test set after feature selection and model selection are complete.

## 10. Performance Evaluation Metrics

The notebook compares the baseline and proposed models using the following metrics.

### Accuracy

Measures the proportion of correctly classified test images.

### Macro Precision

Calculates precision separately for each class and averages the class scores equally.

### Macro Recall

Calculates recall separately for each class and averages the class scores equally.

### Macro-F1

Calculates the F1 score for each class and takes the unweighted average. This is useful when class performance needs to be considered equally.

### Weighted-F1

Calculates the class-wise F1 scores and averages them according to the number of true samples in each class.

### Balanced Accuracy

Calculates the average recall across classes.

### Macro Specificity

Measures the average class-wise specificity using a one-versus-rest confusion-matrix approach.

### Cohen's Kappa

Measures agreement between predicted and true labels while accounting for agreement expected by chance.

### Matthews Correlation Coefficient

Provides a classification-performance measure that considers true positives, true negatives, false positives, and false negatives across the multiclass problem.

### Computational Time

Records the time required for GA search, model fitting, and test prediction, where applicable.

## 11. Visualizations

The notebook generates the following plots and outputs:

- Example medical images from the dataset.
- GA fitness across generations for both configurations.
- Comparison of the number of HOG features before and after selection.
- A performance comparison table for the baseline and proposed models.
- Classification reports for both approaches.

These outputs help assess whether feature selection improves classification performance and reduces the feature space.

## 12. Experimental Results

The notebook generates actual experimental results when executed.

The main comparison includes:

| Metric | HOG + SVM | GA-HOG + SVM |
|---|---|---|
| Accuracy | Generated during execution | Generated during execution |
| Macro Precision | Generated during execution | Generated during execution |
| Macro Recall | Generated during execution | Generated during execution |
| Macro-F1 | Generated during execution | Generated during execution |
| Weighted-F1 | Generated during execution | Generated during execution |
| Balanced Accuracy | Generated during execution | Generated during execution |
| Macro Specificity | Generated during execution | Generated during execution |
| Cohen's Kappa | Generated during execution | Generated during execution |
| Matthews Correlation Coefficient | Generated during execution | Generated during execution |
| Test Prediction Time | Generated during execution | Generated during execution |
| Number of Features | All HOG features | GA-selected features |

The actual values depend on the dataset contents, sampling configuration, random seeds, software environment, and available hardware. No performance improvement should be claimed until the experimental results have been examined.

## 13. Strengths of the Proposed Method

- The GA searches for a subset of HOG features rather than automatically using every extracted feature.
- The feature-selection penalty encourages lower dimensionality.
- Validation macro-F1 allows classification performance to be considered across classes.
- Two GA configurations allow an initial investigation of parameter sensitivity.
- Identical data splits support a fair comparison between the baseline and proposed models.
- Multiple metrics provide a broader assessment than accuracy alone.

## 14. Limitations

- Genetic Algorithms may require substantial computation because fitness evaluation repeatedly trains SVM classifiers.
- Results may vary with population size, mutation probability, number of generations, random seed, and feature-selection penalty.
- Limiting the number of images per class reduces runtime but may affect generalization and the representativeness of results.
- HOG is a hand-crafted feature extractor and may not represent every important characteristic of medical images.
- Feature selection may reduce dimensionality without improving classification performance.
- Results from one dataset may not generalize to other medical image datasets.

## 15. Computational Cost

The baseline approach extracts HOG features and trains a limited number of SVM candidates to select the best model using validation macro-F1.

The proposed approach adds the computational cost of the Genetic Algorithm. Each chromosome may require training an SVM and evaluating its validation macro-F1.

The computational cost depends on:

- Number of images.
- Number of HOG features.
- GA population size.
- Number of generations.
- Mutation and crossover operations.
- SVM training time.

The notebook records experimental timing so that the classification benefits of GA-based feature selection can be considered alongside its computational overhead.

## 16. Conclusion

This project investigates whether Genetic Algorithm-based feature selection can improve HOG-based medical image classification while reducing feature dimensionality.

The baseline HOG + SVM pipeline uses all extracted HOG features, whereas the proposed GA-HOG + SVM pipeline uses a subset selected through validation-based fitness evaluation.

The experimental comparison considers classification performance, feature reduction, and computational cost. The final conclusion must be based on the actual test results generated by the notebook.

If the GA-HOG model achieves better classification metrics while reducing the number of features, the results support its use for this experiment. If classification performance decreases or remains unchanged, the additional computational cost must be weighed against any dimensionality reduction achieved.

## 17. How to Run the Notebook

1. Download `Lab_07_HOG_Genetic_Algorithm.ipynb`.
2. Open [Google Colab](https://colab.research.google.com/) or Jupyter Notebook.
3. Upload and open the notebook.
4. Run the installation cell to install the required libraries.
5. Execute the Kaggle download cell.
6. If requested, configure Kaggle API authentication using `kaggle.json`.
7. Run the preprocessing cell.
8. Execute the remaining tasks in order.
9. Review the performance tables, plots, and generated discussion.
10. Save the notebook with its outputs for submission.

**Important:** Run all cells in order because later tasks depend on variables and models created in earlier cells.

## 18. Generated Output Files

After the relevant notebook cells execute successfully, the following CSV files are generated:

- `lab07_test_results.csv` — test-set performance comparison.
- `lab07_feature_counts.csv` — feature counts before and after GA selection.
- GA fitness history CSV files — fitness information for each GA configuration.

The notebook also displays the plots and comparison tables needed for the lab report.

## 19. References

1. Medical MNIST Dataset, Kaggle: https://www.kaggle.com/datasets/andrewmvd/medical-mnist
2. scikit-image HOG documentation: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog
3. scikit-learn SVM documentation: https://scikit-learn.org/stable/modules/svm.html
4. scikit-learn classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
5. KaggleHub documentation: https://github.com/Kaggle/kagglehub
