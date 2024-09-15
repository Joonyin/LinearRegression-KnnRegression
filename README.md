# House Price Prediction: Comparing Linear Regression and k-Nearest Neighbors (kNN)

## Overview

This project focuses on predicting house prices using two different machine learning models: Linear Regression and k-Nearest Neighbors (kNN) Regression. The goal is to evaluate the performance of both models in terms of their prediction accuracy, comparing them on metrics such as Root Mean Squared Error (RMSE) and R-squared (R²) score.

The dataset used for this project comes from Kaggle's House Prices - Advanced Regression Techniques competition. This dataset includes various features such as the size of the house, number of bedrooms, year built, and more, which we will use to predict the final sale price.

## Project Workflow
1. Data Preprocessing
Load the dataset from Kaggle.
Handle missing values using simple imputation.
Normalize the numerical features using StandardScaler.
Encode categorical variables using OneHotEncoder.
Engineer new features, such as calculating the age of the house.
2. Model Training
Linear Regression: A basic regression algorithm that assumes a linear relationship between the independent variables and the dependent variable.
k-Nearest Neighbors (kNN) Regression: A non-parametric algorithm that makes predictions based on the average of the 'k' nearest data points in the feature space.
3. Model Evaluation
We use RMSE and R² score as evaluation metrics to measure model performance on the test set. The results from both models are compared to determine which algorithm performs better in predicting house prices.

## Results
k-Nearest Neighbors (kNN)
RMSE: 35,308.64
R² Score: 0.82134
Knn Regression provided a relatively good fit for the data, with a lower RMSE and higher R² score, indicating that it is able to explain a good portion of the variance in the target variable (house prices).

Linear Regression
RMSE: 42,645.12
R² Score: 0.73938
Linear Regression performed slightly worse than Knn regression. It had a higher RMSE, meaning its predictions were less accurate, and a lower R² score, meaning it explained less of the variance in the data.

Best Parameters for kNN:
n_neighbors: 9
weights: 'distance'
A grid search was performed to optimize the hyperparameters for the kNN model. The best results were obtained using 9 neighbors and the 'distance' weighting scheme.

## Conclusion
Based on the results, kNN Regression outperforms Linear Regression in this scenario, providing more accurate predictions with a lower RMSE and a higher R² score. This indicates that the Knn model is better suited for this particular dataset.

### Dataset
The dataset used in this project is available from Kaggle: House Prices - Advanced Regression Techniques.
