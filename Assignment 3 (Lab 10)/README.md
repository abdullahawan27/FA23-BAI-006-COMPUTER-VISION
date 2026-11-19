# Machine Learning Regression: California Housing Price Prediction

## 1. Introduction

This project builds and compares multiple machine learning regression models to predict median house values using the California Housing dataset. The notebook follows a complete workflow, from loading and preparing the data to training models, evaluating their performance, and identifying the best-performing algorithm.

## 2. Dataset

The project uses the **California Housing dataset** provided by `scikit-learn` through `fetch_california_housing()`.

- **Input features:** 8
- **Target variable:** `MedHouseVal` (median house value)
- **Dataset size:** 20,640 samples
- **Training data:** 80%
- **Testing data:** 20%

The eight input features are:

1. `MedInc` – Median income
2. `HouseAge` – Median house age
3. `AveRooms` – Average number of rooms
4. `AveBedrms` – Average number of bedrooms
5. `Population` – Population
6. `AveOccup` – Average household occupancy
7. `Latitude` – Geographic latitude
8. `Longitude` – Geographic longitude

## 3. Project Workflow

The notebook performs the following steps:

1. Import the required Python libraries.
2. Load the California Housing dataset.
3. Separate the input features (`X`) and target variable (`y`).
4. Display the dataset shape and sample records.
5. Check the input features for missing values.
6. Split the dataset into training and testing sets using an 80:20 ratio.
7. Train six regression algorithms.
8. Generate predictions for the test data.
9. Evaluate and compare the models using regression metrics.
10. Plot actual house values against predicted values.
11. Identify the model with the highest R² score and the model with the lowest RMSE.

## 4. Machine Learning Algorithms

The following six algorithms are implemented:

| Algorithm | Description |
|---|---|
| Linear Regression | Models the relationship between input features and the target using a linear equation. |
| Decision Tree Regressor | Predicts values using decision rules arranged in a tree structure. |
| Random Forest Regressor | Combines predictions from multiple decision trees. |
| Neural Network (MLPRegressor) | Learns patterns using interconnected layers of artificial neurons. |
| XGBoost Regressor | Uses gradient boosting to build an ensemble of decision trees. |
| K-Nearest Neighbors (KNN) Regressor | Predicts a value using the target values of the nearest training samples. |

Feature scaling with `StandardScaler` is used in pipelines for Linear Regression, Neural Network, and KNN. The Decision Tree, Random Forest, and XGBoost models are trained without a scaling step.

## 5. Model Evaluation

Each model is evaluated using four regression metrics:

- **MAE (Mean Absolute Error):** The average absolute difference between actual and predicted values. Lower is better.
- **MSE (Mean Squared Error):** The average squared prediction error. Lower is better.
- **RMSE (Root Mean Squared Error):** The square root of MSE. Lower is better.
- **R² Score:** Indicates how much of the variation in the target is explained by the model. A higher score is generally better.

The notebook creates a comparative results table containing all four metrics for each algorithm.

## 6. Visualizations

A set of six scatter plots compares actual house values with predicted house values, with one plot for each model. A dashed reference line represents perfect predictions. Points closer to this line indicate predictions closer to the actual values.

## 7. Best Model Selection

At the end of the notebook:

- The algorithm with the **highest R² score** is identified.
- The algorithm with the **lowest RMSE** is identified.

These results help compare the models from two complementary perspectives. The notebook calculates the results when it is run; the winning algorithm and metric values may vary with the execution environment or library versions.

## 8. Requirements

Install the required libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

## 9. How to Run

1. Open `Revised_ML.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Ensure the required libraries are installed.
3. Run the notebook cells in order, from the first cell to the last.
4. Review the printed evaluation table, scatter plots, and best-model results.

The California Housing dataset is fetched through scikit-learn, so an internet connection may be required the first time it is downloaded.

## 10. Conclusion

This project demonstrates how different regression algorithms can be applied to the same dataset and compared fairly using a separate test set. It evaluates six models with MAE, MSE, RMSE, and R², and uses a results table and visualizations to examine their prediction performance.
