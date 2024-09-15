##House Price Prediction: Comparing Linear Regression and k-Nearest Neighbors (kNN)
#Overview
This project focuses on predicting house prices using two different machine learning models: Linear Regression and k-Nearest Neighbors (kNN) Regression. The goal is to evaluate the performance of both models in terms of their prediction accuracy, comparing them on metrics such as Root Mean Squared Error (RMSE) and R-squared (R²) score.

The dataset used for this project comes from Kaggle's House Prices - Advanced Regression Techniques competition. This dataset includes various features such as the size of the house, number of bedrooms, year built, and more, which we will use to predict the final sale price.

#Project Workflow
Data Preprocessing:

Load the dataset from Kaggle.
Handle missing values using simple imputation.
Normalize the numerical features using StandardScaler.
Encode categorical variables using OneHotEncoder.
Engineer new features, such as calculating the age of the house.
Model Training:

Linear Regression: This is a basic regression algorithm that assumes a linear relationship between the independent variables and the dependent variable.
k-Nearest Neighbors (kNN) Regression: This is a non-parametric algorithm that makes predictions based on the average of the 'k' nearest data points in the feature space.
Model Evaluation:

Use RMSE and R² score as evaluation metrics to measure model performance on the test set.
Compare the results from both models to determine which algorithm performs better in predicting house prices.
#Results
Linear Regression:
RMSE: 35,308.64
R² Score: 0.82134
Linear regression provided a relatively good fit for the data, with a lower RMSE and higher R² score, indicating that it is able to explain a good portion of the variance in the target variable (house prices).
k-Nearest Neighbors (kNN):
RMSE: 42,645.12
R² Score: 0.73938
The kNN model performed slightly worse than linear regression. It had a higher RMSE, meaning its predictions were less accurate, and a lower R² score, meaning it explained less of the variance in the data.
Best Parameters for kNN:
n_neighbors: 9
weights: 'distance'
A grid search was performed to optimize the hyperparameters for the kNN model. The best results were obtained using 9 neighbors and the 'distance' weighting scheme.
Conclusion
Based on the results, Linear Regression outperforms kNN Regression in this scenario, providing more accurate predictions with a lower RMSE and a higher R² score. This indicates that the linear model is better suited for this particular dataset.

#Dataset
The dataset used in this project is available from Kaggle: House Prices - Advanced Regression Techniques Dataset.
