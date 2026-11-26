# Deep Learning & Computer Vision Lab
## CNN Feature Visualization, Parameter Audit & Critical Analysis

**Student Name:** M. Abdullah Awan 
**Registration No.:** FA23-BAI-006  


---

## 1. Project Overview

This project completes the practical laboratory assignment **“CNN Feature Visualization, Parameter Audit & Critical Analysis”**. It demonstrates how convolutional filters transform an input image into feature maps and how activation and pooling operations change those maps. It also builds and compares two lightweight Convolutional Neural Network (CNN) architectures and verifies their trainable parameter counts using both manual calculations and TensorFlow/Keras.

The notebook is designed to run in **Google Colab** or a local **Jupyter Notebook** environment.

### Objectives

1. Define five custom 3×3 image-processing kernels.
2. Apply each kernel independently to a grayscale image.
3. Record four pipeline stages: raw convolution, ReLU, max pooling, and average pooling.
4. Produce a single 5 × 4 visualization grid.
5. Build and compare two CNN architectures for 32 × 32 × 3 RGB input.
6. Calculate trainable parameters manually and verify them with Keras model summaries.
7. Answer all five critical-thinking and viva questions from the assignment.

---

## 2. Notebook Contents

The notebook is organized into the following sections:

| Section | Description |
|---|---|
| 1 | Import libraries and set random seeds |
| 2 | Generate or upload an input image and prepare it as grayscale |
| 3 | Define five custom 3×3 kernels |
| 4 | Execute convolution, ReLU, max pooling, and average pooling |
| 5 | Display the 5 × 4 feature-map visualization grid |
| 6 | Calculate numerical statistics for each feature map |
| 7 | Build Model A and Model B and perform the parameter audit |
| 8 | Answer the five critical-thinking and viva questions |
| 9 | Review the final assignment checklist |

---

## 3. Technologies and Libraries

The notebook uses:

- **Python** — primary programming language.
- **NumPy** — numerical arrays, image data, and kernel definitions.
- **Matplotlib** — input-image display and feature-map visualizations.
- **Pandas** — tabular presentation of feature-map statistics and parameter counts.
- **TensorFlow** — convolution, activation, pooling, and CNN implementation.
- **Keras (`tensorflow.keras`)** — construction and inspection of the two CNN models.
- **Pillow (`PIL`)** — opening, converting, and resizing an uploaded image.

### Installation

Google Colab normally includes the required libraries. If running locally, install the dependencies with:

```bash
pip install numpy matplotlib pandas tensorflow pillow nbformat
```

---

## 4. Input Image Preparation

The assignment requires a clear grayscale or RGB image containing geometric edges, textures, and contrast gradients.

By default, the notebook generates a reproducible **256 × 256 architectural-style grayscale image**. It contains:

- Horizontal structural beams.
- Vertical structural beams.
- Repeated window-like regions.
- Light and dark intensity areas.
- A small amount of deterministic noise and a gentle intensity gradient.

This image provides clear horizontal and vertical boundaries for testing the filters.

### Using a personal image

The notebook includes an optional upload path. Set:

```python
USE_UPLOADED_IMAGE = True
```

When running in Google Colab, the notebook opens the upload dialog. Select an image from your computer. The uploaded image is converted to grayscale and resized to **256 × 256** pixels.

To use the generated example image instead, keep:

```python
USE_UPLOADED_IMAGE = False
```

The random seed is set to 42 to make the generated example and model initialization reproducible.

---

## 5. Part I — Five Custom 3×3 Kernels

Five filters are explicitly defined and applied independently.

### 5.1 Horizontal Edge Detector — Sobel-X

```text
[[-1,  0,  1],
 [-2,  0,  2],
 [-1,  0,  1]]
```

This kernel responds strongly to left-to-right intensity changes and detects vertical boundaries in the image.

### 5.2 Vertical Edge Detector — Sobel-Y

```text
[[-1, -2, -1],
 [ 0,  0,  0],
 [ 1,  2,  1]]
```

This kernel responds strongly to top-to-bottom intensity changes and detects horizontal boundaries in the image.

### 5.3 Sharpen Kernel

```text
[[ 0, -1,  0],
 [-1,  5, -1],
 [ 0, -1,  0]]
```

The sharpen filter emphasizes local intensity differences and makes edges and fine details more prominent.

### 5.4 Box Blur Kernel

```text
[[1/9, 1/9, 1/9],
 [1/9, 1/9, 1/9],
 [1/9, 1/9, 1/9]]
```

The box blur averages neighboring pixel values. It smooths the image and reduces small variations and noise.

### 5.5 Emboss Kernel

```text
[[-2, -1, 0],
 [-1,  1, 1],
 [ 0,  1, 2]]
```

The emboss filter creates a raised or engraved appearance by emphasizing directional intensity transitions.

---

## 6. Sequential Filter Pipeline

Each of the five filters is applied independently to the same grayscale input image. The notebook records four stages for every filter.

### Stage A — Raw Convolution Output

The input image is convolved with the selected 3×3 kernel. This is the direct linear feature map before applying an activation function.

The convolution operation can produce positive, negative, and near-zero values.

### Stage B — Post-ReLU Activation

The ReLU activation function is applied:

\[
\operatorname{ReLU}(x)=\max(0,x)
\]

- Positive values remain unchanged.
- Negative values become zero.
- Zero values remain zero.

This removes negative responses and introduces non-linearity.

### Stage C — Max Pooling

Max pooling uses:

- Pool size: **2 × 2**
- Stride: **2**
- Padding: **VALID**

It selects the largest value from each 2×2 region. This retains strong positive activations while reducing the spatial dimensions.

### Stage D — Average Pooling

Average pooling also uses:

- Pool size: **2 × 2**
- Stride: **2**
- Padding: **VALID**

It calculates the average of the four values in each 2×2 region. This produces a smoother, downsampled feature map.

### Implementation details

- Convolution is performed using `tf.nn.conv2d`.
- Convolution stride is 1.
- Convolution padding is `SAME`.
- ReLU is performed using `tf.nn.relu`.
- Max pooling is performed using `tf.nn.max_pool2d`.
- Average pooling is performed using `tf.nn.avg_pool2d`.

The pooling stages are both applied to the **Post-ReLU feature map**, as required by the assignment.

---

## 7. Required 5 × 4 Visualization Grid

The notebook creates one unified visualization containing **5 rows and 4 columns**.

### Rows — Five filters

1. Horizontal Edge (Sobel-X)
2. Vertical Edge (Sobel-Y)
3. Sharpen
4. Box Blur
5. Emboss

### Columns — Four processing stages

1. Stage A — Raw Convolution
2. Stage B — Post-ReLU
3. Stage C — Max Pooling
4. Stage D — Average Pooling

Each subplot has a title and an intensity colorbar.

The raw convolution outputs use a diverging `seismic` color map with a symmetric intensity range, allowing positive and negative responses to be distinguished. The remaining stages use grayscale display.

The grid makes it possible to compare how each filter responds to the same image and how ReLU and pooling modify its output.

---

## 8. Numerical Feature-Map Statistics

In addition to visualizing the feature maps, the notebook calculates the following values for every filter at every stage:

- **Minimum:** smallest value in the feature map.
- **Maximum:** largest value in the feature map.
- **Mean:** average value of all feature-map elements.
- **Standard deviation:** amount of variation in the feature-map values.
- **Shape:** dimensions of the feature map.

These values are displayed in a Pandas DataFrame. They provide numerical evidence to support observations made from the visualization grid.

---

## 9. Part II — CNN Architectural Comparison

The assignment specifies two lightweight CNN models with:

- Input size: **32 × 32 × 3 (RGB)**
- First convolution layer: **5 filters**
- Second convolution layer: **10 filters**
- Pooling size: **2 × 2**
- Pooling stride: **2**
- Global Average Pooling
- Dense output layer: **10 classes**

Each convolution layer uses ReLU activation and `padding="same"`. The specified pooling operation is applied after each convolution layer.

### 9.1 Model A — Standard CNN

| Layer | Configuration |
|---|---|
| Input | 32 × 32 × 3 |
| Conv1 | 5 filters, 3 × 3 kernel, ReLU |
| Pool1 | Max Pooling, 2 × 2, stride 2 |
| Conv2 | 10 filters, 3 × 3 kernel, ReLU |
| Pool2 | Max Pooling, 2 × 2, stride 2 |
| GAP | Global Average Pooling |
| Output | Dense, 10 units, Softmax |

Model A uses standard 3×3 convolution kernels and max pooling.

### 9.2 Model B — Large-Kernel CNN

| Layer | Configuration |
|---|---|
| Input | 32 × 32 × 3 |
| Conv1 | 5 filters, 5 × 5 kernel, ReLU |
| Pool1 | Average Pooling, 2 × 2, stride 2 |
| Conv2 | 10 filters, 5 × 5 kernel, ReLU |
| Pool2 | Average Pooling, 2 × 2, stride 2 |
| GAP | Global Average Pooling |
| Output | Dense, 10 units, Softmax |

Model B uses larger 5×5 convolution kernels and average pooling.

**Note:** The notebook builds and summarizes these architectures. It does not train them on a labeled dataset because the assignment asks for architecture construction and parameter auditing, not a classification training experiment.

---

## 10. Manual Trainable Parameter Calculation

The assignment provides the convolution parameter formula:

\[
P=(K_h\times K_w\times C_{in}+1)\times C_{out}
\]

Where:

- \(K_h\) = kernel height.
- \(K_w\) = kernel width.
- \(C_{in}\) = number of input channels.
- \(C_{out}\) = number of output filters.
- The additional 1 represents one bias per output filter.

Pooling and Global Average Pooling layers have no trainable parameters.

### 10.1 Model A — 3×3 Kernels

**Conv1**

Input channels = 3, output filters = 5.

\[
(3\times3\times3+1)\times5
=(27+1)\times5
=140
\]

**Conv2**

Input channels = 5, output filters = 10.

\[
(3\times3\times5+1)\times10
=(45+1)\times10
=460
\]

Convolution subtotal:

\[
140+460=600
\]

**Dense output layer**

Global Average Pooling produces 10 features, one for each final feature channel.

\[
(10+1)\times10=110
\]

**Total Model A parameters**

\[
600+110=\boxed{710}
\]

### 10.2 Model B — 5×5 Kernels

**Conv1**

\[
(5\times5\times3+1)\times5
=(75+1)\times5
=380
\]

**Conv2**

\[
(5\times5\times5+1)\times10
=(125+1)\times10
=1260
\]

Convolution subtotal:

\[
380+1260=1640
\]

**Dense output layer**

\[
(10+1)\times10=110
\]

**Total Model B parameters**

\[
1640+110=\boxed{1750}
\]

### 10.3 Parameter Audit Summary

| Model | Conv1 | Conv2 | Dense | Total Trainable Parameters |
|---|---:|---:|---:|---:|
| Model A — Standard | 140 | 460 | 110 | **710** |
| Model B — Large Kernel | 380 | 1260 | 110 | **1750** |

The notebook verifies these manual totals against the trainable parameter counts reported by `model.summary()` and Keras model weights.

### 10.4 Percentage Increase

For the complete models:

\[
\frac{1750-710}{710}\times100
=\boxed{146.48\%}
\]

Therefore, Model B has **146.48% more total trainable parameters** than Model A.

If only the convolution layers are compared, excluding the identical Dense classifier:

\[
\frac{1640-600}{600}\times100
=\boxed{173.33\%}
\]

Both percentages are reported to make clear whether the comparison includes the classifier.

---

## 11. Part III — Critical Thinking and Viva Defense

The notebook includes answers to all five questions from the assignment.

### Question 1 — ReLU “Blackout” Effect and Activation Alternatives

Negative convolution responses become zero after ReLU. In the horizontal and vertical edge maps, these black regions correspond to local intensity transitions whose polarity produces negative responses for the selected kernel. Flat areas can also have responses close to zero.

ReLU is:

\[
f(x)=\max(0,x)
\]

**Leaky ReLU** keeps negative values at a reduced scale:

\[
f(x)=
\begin{cases}
x & x\geq0\\
0.1x & x<0
\end{cases}
\]

Therefore, negative responses remain visible at one tenth of their original magnitude.

**Sigmoid** is:

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

It maps values into the range (0,1). Negative values become values below 0.5, while large positive and negative values approach 1 and 0 respectively.

### Question 2 — Max Pooling vs. Average Pooling

Max pooling selects the largest activation in each 2×2 region. It can preserve a strong positive edge or sharpened response when that response is the largest value in its local region.

Average pooling calculates the mean of the four values. A single strong activation can become weaker when averaged with lower neighboring values.

For dark ink on a pure white background, ordinary grayscale max pooling tends to select the high-valued white background and can lose thin dark strokes. Average pooling generally retains some influence from the dark pixels. If the image is inverted so ink is represented by high values, max pooling can preserve strong ink activations more effectively.

### Question 3 — Parameter Explosion and Receptive Field Equivalence

Convolution-only parameter counts:

- Model A: 600.
- Model B: 1640.

Convolution-only increase:

\[
\frac{1640-600}{600}\times100=173.33\%
\]

Complete model parameter counts, including the Dense classifier:

- Model A: 710.
- Model B: 1750.

Complete model increase:

\[
\frac{1750-710}{710}\times100=146.48\%
\]

Two stride-1 3×3 convolution layers have an effective receptive field of 5×5, assuming no pooling between them.

For a comparison using 5 input channels, 5 intermediate channels, and 10 output channels:

One 5×5 convolution:

\[
(5\times5\times5+1)\times10=1260
\]

Two 3×3 convolutions:

\[
(3\times3\times5+1)\times5=230
\]

\[
(3\times3\times5+1)\times10=460
\]

Total:

\[
230+460=690
\]

Parameters saved:

\[
1260-690=\boxed{570}
\]

This is a 45.24% reduction relative to the single 5×5 layer. The saving depends on the selected intermediate channel count.

### Question 4 — Operation Commutativity

For max pooling and ReLU, the output is identical whether ReLU is applied before or after max pooling:

\[
\max(\operatorname{ReLU}(x_i))
=\operatorname{ReLU}(\max(x_i))
\]

Example 2×2 patch:

\[
\begin{bmatrix}
-4 & 2\\
-1 & -3
\end{bmatrix}
\]

**Conv → ReLU → MaxPool**

After ReLU:

\[
\begin{bmatrix}
0 & 2\\
0 & 0
\end{bmatrix}
\]

Max pooling gives 2.

**Conv → MaxPool → ReLU**

Max pooling first gives 2. ReLU then gives 2.

Both orders produce the same result because ReLU is monotonic and max pooling selects the largest value. This equivalence is specific to max pooling and ReLU; it should not be assumed for average pooling.

### Question 5 — Diagnosing Filter Redundancy

If all five first-layer filters produce nearly identical feature maps, the network is not learning diverse feature detectors.

**Cause 1: Identical or overly similar weight initialization**

If filters begin with the same weights, they may receive similar gradients and learn similar features.

**Fix:** Use independent random initialization, such as He initialization for ReLU-based networks:

```python
kernel_initializer="he_normal"
```

**Cause 2: Optimization or gradient problems**

An unsuitable learning rate can prevent useful learning. Dead ReLU units can also stop receiving gradients when their inputs remain negative.

**Fixes:**
- Inspect activation distributions and gradient norms.
- Start with a reasonable learning rate, such as Adam with 0.001, and tune it using validation data.
- Consider Leaky ReLU if many units are inactive.

Other checks include verifying input normalization, labels, and that filters are not accidentally tied or shared.

---

## 12. Expected Notebook Outputs

After running all cells, the notebook displays:

1. NumPy and TensorFlow version information.
2. The selected 256 × 256 grayscale input image.
3. All five 3×3 kernel matrices.
4. The shapes of all intermediate feature maps.
5. The complete 5 × 4 filter visualization grid with colorbars.
6. A feature-map statistics DataFrame containing minimum, maximum, mean, standard deviation, and shape.
7. The manual convolution parameter audit table.
8. The Model A Keras summary.
9. The Model B Keras summary.
10. A complete parameter comparison table showing manual counts, Keras counts, and whether they match.
11. The five critical-thinking answers in the notebook.

The exact pixel values and feature-map statistics depend on the selected input image. The default generated image is deterministic because a fixed random seed is used.

---

## 13. How to Run the Notebook in Google Colab

1. Download `CNN_Visualization_Parameter_Audit_Lab.ipynb`.
2. Open [Google Colab](https://colab.research.google.com/).
3. Select **File → Upload notebook**.
4. Upload the downloaded `.ipynb` file.
5. Keep `USE_UPLOADED_IMAGE = False` to use the generated example image, or change it to `True` to upload your own image.
6. Select **Runtime → Run all**.
7. Wait for every cell to finish.
8. Review the visualization grid, statistics, model summaries, and critical-thinking answers.

A GPU is not required for this small demonstration. A standard CPU runtime is sufficient.

---

## 14. Submission Evidence and Screenshots

The assignment rubric requires visual evidence and a successful viva defense. After running the notebook, capture clear screenshots of:

1. The input grayscale image.
2. The five custom kernel definitions or printed kernel matrices.
3. The complete 5 × 4 visualization grid.
4. The feature-map statistics table.
5. Model A's `model.summary()` output.
6. Model B's `model.summary()` output.
7. The manual and framework parameter comparison table.

Keep the screenshots readable and ensure the complete grid and parameter values are visible.

### Assignment grading rubric

| Assessment Criterion | Weight | What is evaluated |
|---|---:|---|
| Visualization Grid & Code Execution | 20% | Reproducible 5 × 4 visualization with labeled filters, stages, and intensity scales |
| Manual Parameter Audit | 25% | Correct step-by-step calculations verified against framework summaries |
| Critical Analysis & Viva Defense | 55% | Mathematically sound answers grounded in the generated feature maps |

---

## 15. Important Notes and Assumptions

- The notebook follows the assignment's five-filter, four-stage pipeline.
- Each filter is applied independently to the same grayscale input image.
- Convolution uses `SAME` padding and stride 1.
- Pooling uses 2×2 windows, stride 2, and `VALID` padding.
- Both max and average pooling operate on the Post-ReLU feature map in Part I.
- The CNN input shape is 32 × 32 × 3 RGB.
- Model A uses 3×3 kernels and max pooling.
- Model B uses 5×5 kernels and average pooling.
- Both CNNs use 5 filters in Conv1, 10 filters in Conv2, GAP, and a 10-unit softmax output.
- The model parameter audit includes convolution layers and the Dense classifier. Pooling and GAP have no trainable parameters.
- The CNNs are constructed and summarized but are not trained because model training is not required by the supplied assignment.
- Exact visual observations should be checked against the feature maps produced by the image actually used.

---

## 16. Conclusion

This laboratory demonstrates the basic feature-extraction stages used in CNNs. Five custom filters are applied to an image to show how edge detection, sharpening, blurring, and embossing affect image information. ReLU removes negative responses, while max pooling and average pooling downsample feature maps in different ways.

Two CNN architectures are then constructed and their trainable parameters are calculated manually and verified using TensorFlow/Keras. The results show that larger 5×5 kernels require more parameters than 3×3 kernels in the specified architectures. Finally, the five critical-thinking questions explain activation behavior, pooling information retention, receptive fields, operation ordering, and filter redundancy.

**Final verified parameter totals:**
- **Model A:** 710 trainable parameters.
- **Model B:** 1,750 trainable parameters.
- **Complete-model parameter increase:** 146.48%.

Run the notebook from top to bottom and use the generated outputs as evidence when preparing the lab submission and viva.
