# SIFT Landmark Recognition Pipeline

## Overview
This project provides an end-to-end computer vision notebook pipeline designed to construct a reference landmark image database, extract Scale-Invariant Feature Transform (SIFT) descriptors, match features using Brute-Force Matchers with Lowe's ratio test, and verify spatial consistency via RANSAC geometric filtering. It is optimized for execution in cloud environments such as Google Colab[cite: 1] or local Jupyter Notebook servers.

---

## Features
* **Automated Image Acquisition**: Programmatically fetches and filters reference image datasets for prominent international landmarks directly from Wikimedia Commons[cite: 1].
* **Robust Feature Extraction**: Employs OpenCV's SIFT algorithm (`cv2.SIFT_create`) to detect illumination-, scale-, and rotation-invariant keypoints[cite: 1].
* **Advanced Descriptor Matching**: Implements $k$-NN feature matching paired with Lowe's ratio test thresholding to eliminate ambiguous matches[cite: 1].
* **Geometric Verification**: Applies RANSAC (RANdom SAmple Consensus) homography estimation to filter out outliers and confirm true landmark matches.
* **Result Visualization**: Generates side-by-side matching plots showing keypoint correspondences and inlier matches using Matplotlib[cite: 1].

---

## Requirements & Dependencies
The notebook automatically handles package installation, but the execution requires the following core dependencies:
* Python 3.x[cite: 1]
* `opencv-contrib-python-headless` (OpenCV build featuring patent-free SIFT support)[cite: 1]
* `requests`[cite: 1]
* `pandas`[cite: 1]
* `matplotlib`[cite: 1]
* `numpy`[cite: 1]

---

## Project Structure
During execution, the pipeline dynamically organizes downloaded reference files into a localized hierarchical directory tree:

```text
landmark_database/
    Eiffel_Tower/
        reference_01.jpg
        reference_02.jpg
        reference_03.jpg
    Taj_Mahal/
        reference_01.jpg
        reference_02.jpg
    Great_Wall_of_China/
        ...
    Statue_of_Liberty/
        ...
    Colosseum/
        ...