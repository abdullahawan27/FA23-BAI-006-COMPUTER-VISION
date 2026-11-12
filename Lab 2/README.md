# Skin Cancer Classification Using Deep Learning

## Overview

This project focuses on **multi-class skin cancer image classification** using deep learning and transfer learning techniques.

The project evaluates multiple pre-trained CNN architectures and identifies high-performing models for further experimentation. The selected models — **ResNet101, ResNet50, and ResNet18** — are then evaluated with different image-processing filters to study their effect on classification performance.

## Objectives

* Compare different transfer learning models for skin cancer classification.
* Evaluate classification performance using multiple metrics.
* Analyze computational efficiency of different CNN architectures.
* Select high-performing models for further experimentation.
* Apply different image filters and evaluate their impact on model performance.

## Dataset

The project uses the **Skin Cancer 9 Classes ISIC** dataset downloaded through KaggleHub.

The dataset contains **9 skin lesion classes**:

* Actinic Keratosis
* Basal Cell Carcinoma
* Dermatofibroma
* Melanoma
* Nevus
* Pigmented Benign Keratosis
* Seborrheic Keratosis
* Squamous Cell Carcinoma
* Vascular Lesion

A total of **2,357 valid images** were detected and used in the experiment. The data was divided using an **80/20 stratified train-validation split**.

## Technologies Used

* Python
* PyTorch
* Torchvision
* OpenCV
* Scikit-learn
* XGBoost
* NumPy
* Pandas
* Pillow
* KaggleHub
* THOP
* Google Colab
* CUDA / GPU

## Models Evaluated

The initial experiment evaluated the following transfer learning architectures:

* AlexNet
* VGG16
* VGG19
* ResNet18
* ResNet50
* ResNet101
* DenseNet121
* EfficientNet-B0

The models were initialized with pre-trained ImageNet weights and their final classification layers were modified for the 9-class skin cancer classification task.

## Initial Model Results

The initial transfer-learning experiment produced the following accuracy results:

| Model           | Accuracy |
| --------------- | -------: |
| AlexNet         |   64.19% |
| VGG16           |   55.72% |
| VGG19           |   63.35% |
| ResNet18        |   68.01% |
| ResNet50        |   68.86% |
| ResNet101       |   69.92% |
| DenseNet121     |   67.37% |
| EfficientNet-B0 |   64.19% |

Based on these results, **ResNet101, ResNet50, and ResNet18** were selected for the image-filtering experiment.

## Image Filtering Experiment

Six input conditions were tested for each selected model:

1. No Filter
2. Average Filter
3. Gaussian Filter
4. Median Filter
5. Sharpening Filter
6. Sobel Filter

The filters were implemented using OpenCV. Average, Gaussian, and Median filters perform smoothing operations, while Sharpening enhances image details and Sobel emphasizes image edges.

Each filtered image was resized to **224 × 224 pixels** and normalized using the standard ImageNet mean and standard deviation values before being passed to the CNN models.

## Filtering Experiment Results

| Model     | Filter     | Accuracy | Precision | Recall | F1-Score | Macro-F1 |    AUC |
| --------- | ---------- | -------: | --------: | -----: | -------: | -------: | -----: |
| ResNet101 | No Filter  |   69.28% |    67.17% | 69.28% |   67.43% |   58.52% | 93.28% |
| ResNet101 | Average    |   70.34% |    68.77% | 70.34% |   68.26% |   60.13% | 93.98% |
| ResNet101 | Gaussian   |   68.64% |    66.01% | 68.64% |   66.56% |   57.06% | 92.57% |
| ResNet101 | Median     |   69.92% |    67.81% | 69.92% |   68.27% |   59.11% | 93.56% |
| ResNet101 | Sharpening |   72.03% |    70.72% | 72.03% |   70.16% |   58.85% | 93.85% |
| ResNet101 | Sobel      |   47.88% |    43.29% | 47.88% |   41.45% |   30.31% | 85.78% |
| ResNet50  | No Filter  |   70.13% |    69.24% | 70.13% |   69.03% |   59.90% | 94.01% |
| ResNet50  | Average    |   66.74% |    65.96% | 66.74% |   64.62% |   54.15% | 93.83% |
| ResNet50  | Gaussian   |   69.49% |    68.21% | 69.49% |   68.02% |   58.79% | 93.20% |
| ResNet50  | Median     |   70.55% |    68.42% | 70.55% |   68.35% |   56.84% | 93.93% |
| ResNet50  | Sharpening |   69.28% |    65.67% | 69.28% |   66.02% |   54.06% | 94.08% |
| ResNet50  | Sobel      |   43.64% |    41.85% | 43.64% |   38.85% |   29.47% | 78.37% |
| ResNet18  | No Filter  |   68.86% |    68.35% | 68.86% |   67.54% |   59.17% | 94.69% |
| ResNet18  | Average    |   68.01% |    67.50% | 68.01% |   66.52% |   57.57% | 94.86% |
| ResNet18  | Gaussian   |   66.31% |    65.82% | 66.31% |   65.50% |   56.49% | 94.27% |
| ResNet18  | Median     |   69.28% |    67.98% | 69.28% |   67.86% |   58.24% | 94.80% |
| ResNet18  | Sharpening |   68.64% |    68.16% | 68.64% |   68.03% |   59.10% | 94.31% |
| ResNet18  | Sobel      |   49.79% |    50.14% | 49.79% |   49.24% |   42.68% | 85.45% |

The filtering experiment results are generated directly in the notebook after training and evaluating each selected model under all six filtering conditions.

## Evaluation Metrics

The models are evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Macro-F1**
* **AUC**

The evaluation uses weighted metrics for precision, recall, and F1-score, while Macro-F1 is additionally calculated to evaluate performance across the individual classes.

## Methodology

The overall workflow is:

```text
Dataset
   ↓
Image Loading
   ↓
9-Class Classification
   ↓
80/20 Stratified Train-Validation Split
   ↓
Transfer Learning
   ↓
Model Comparison
   ↓
Select ResNet101, ResNet50, ResNet18
   ↓
Apply Image Filters
   ↓
Train Selected Models
   ↓
Evaluate Performance
   ↓
Compare Results
```

## Training Configuration

The main experiment uses:

* Image size: **224 × 224**
* Batch size: **32**
* Optimizer: **Adam**
* Learning rate: **1e-4**
* Training epochs: **2**
* Train/Validation split: **80/20**
* Random seed: **42**
* GPU acceleration when CUDA is available

These settings are defined in the notebook's training and data-loading pipeline.

## Installation

Install the required Python packages:

```bash
pip install kaggle kagglehub torch torchvision scikit-learn xgboost thop pandas numpy pillow opencv-python
```

## Running the Project

1. Clone the repository.
2. Open the Jupyter Notebook in Google Colab or a Python environment with GPU support.
3. Install the required dependencies.
4. Run the notebook cells sequentially.
5. The dataset is accessed through KaggleHub.
6. The notebook trains and evaluates the selected CNN models.
7. The final results are displayed as a comparison table.

## Repository Structure

```text
Skin-Cancer-Classification/
│
├── skin_cancer.ipynb
├── README.md
└── results/
```

## Project Highlights

* Multi-class classification of skin lesion images.
* Comparison of **8 CNN architectures**.
* Transfer learning using pre-trained ImageNet models.
* Selection of three ResNet architectures for further experimentation.
* Evaluation of **6 different image-filtering conditions**.
* Multiple classification metrics including Accuracy, Precision, Recall, F1-score, Macro-F1, and AUC.
* GPU-accelerated training using PyTorch.

## Disclaimer

This project is intended for **educational and research purposes**. The model outputs should not be used as a substitute for professional medical diagnosis or clinical decision-making.
   

## Question/Answers

## Which three pretrained models performed best in Lab Activity 1?
The three best-performing models were ResNet101 (69.92%), ResNet50 (68.86%), and ResNet18 (68.01%). These were selected for the filtering experiment.
## How does filtering affect each of the three models?
Filtering affected the models differently. Average and sharpening generally improved ResNet101, while ResNet50 and ResNet18 showed smaller or mixed changes across the filters.
## Which filter produces the greatest change compared with the unfiltered baseline?
The Sobel filter produced the greatest change, causing a large decrease in accuracy for all three models. For example, ResNet101 dropped from 69.28% to 47.88%.
## Does the effect of a filter remain consistent across all three models?
No. The same filter does not have the same effect on every model; performance changes depend on the architecture and the features it extracts.
## Does filtering improve or decrease macro-F1 and balanced accuracy?
The effect is mixed. Some filters improve Macro-F1, while others decrease it; for example, ResNet101 Average filtering increased Macro-F1 from 58.52% to 60.13%. Balanced accuracy was included in the code but is not reported in the final results table.
## Which lesion classes are most affected by filtering?
The notebook's final filtering table does not provide class-wise results or confusion matrices, so the specific lesion classes most affected cannot be determined from the available results.
## Why might smoothing remove useful lesion texture or morphological information?
Smoothing reduces high-frequency details and fine textures in lesion images. This can remove boundaries, patterns, and small morphological features that CNNs may use for classification.
## Why might sharpening or edge detection help or hurt classification?
Sharpening can enhance important lesion boundaries and details, potentially helping classification. However, strong edge detection such as Sobel can remove color and texture information, which may explain its substantial performance decrease.
## What is the difference between convolution and correlation?
Correlation slides a kernel over an image without flipping the kernel. Convolution mathematically flips the kernel before applying it; in many deep-learning libraries, the operation called convolution is actually implemented as cross-correlation.
## Relationship between classical image processing and deep-learning feature extraction
Classical filters modify the input image before it reaches the CNN, which can enhance or remove certain features. The results show that preprocessing can influence learned feature extraction, but its effect is model-dependent.