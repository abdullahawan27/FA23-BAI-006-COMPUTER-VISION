
## 11. Discussion Questions

### Q1. Edge Detection and Noise
The detector with the highest noise sensitivity is identified from Table 1 by its measured false/high-density response on noisy input. The Laplacian is generally expected to be strongly affected because it responds to rapid intensity changes, including noise.

### Q2. Effect of Filtering
Gaussian filtering smooths high-frequency variations and generally reduces false edges caused by Gaussian noise. Median filtering preserves stronger boundaries while removing isolated Salt-and-Pepper noise, so it is particularly useful for that noise type.

### Q3. Canny Parameters
Lower thresholds detect more weak edges, which can increase the number of detected edges but may also introduce false edges. Higher thresholds suppress weak responses and usually produce fewer, stronger edges. The exact counts for this dataset are shown in Table 2.

### Q4. Edge Maps and Classification
The answer is determined from the Raw and Edge accuracy values in Table 3. If Edge accuracy is lower, edge-only representation has reduced performance for this experiment; if it is higher, it has improved performance. Edge maps discard information that may be useful for distinguishing visually similar skin-lesion classes.

### Q5. Information Loss
Edge maps mainly preserve boundaries and strong intensity transitions. They can remove or greatly reduce color, texture, local intensity, shading, and other appearance information that may contribute to class discrimination.

### Q6. Classical vs. Deep Features
CNNs can learn edge-like filters automatically and then combine them into higher-level texture and shape features. This allows the network to use both low-level and higher-level information rather than depending only on a manually designed edge representation.

### Q7. Best Representation
The final answer should be taken from the measured Raw, Filtered, and Edge results in Table 3. The representation with the strongest measured classification metrics is the one to report for this experiment.


