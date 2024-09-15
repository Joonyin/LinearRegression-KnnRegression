# AI Singapore
# Regression 2 Exercise
# Exercise: Building a Regression job template
import joblib
# 1. Import required libraries
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import datetime as d


# Information on Data
# https://www.kaggle.com/c/home-data-for-ml-course/data

# Custom Classes and Functions
def display_df_info(df_name, my_df, v=False):
    """Convenience function to display information about a dataframe"""

    print("Data: {}".format(df_name))
    print("Shape (rows, cols) = {}".format(my_df.shape))
    print("First few rows...")
    print(my_df.head())

    # Optional: Display other optional information with the (v)erbose flag
    if v:
        print("Dataframe Info:")
        print(my_df.info())


class GetAge(BaseEstimator, TransformerMixin):
    """Custom Transformer: Calculate age (years only) relative to current year. Note that
    the col values will be replaced but the original col name remains. When the transformer is
    used in a pipeline, this is not an issue as the names are not used. However, if the data
    from the pipeline is to be converted back to a DataFrame, then the col name change should
    be done to reflect the correct data content."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        current_year = int(d.datetime.now().year)
        X['YearBuilt'] = current_year - X['YearBuilt']
        return X


def main():
    # DATA INPUT
    ############
    file_path = "../../data/train.csv"  # Modify this to the correct path of your file
    input_data = pd.read_csv(file_path)
    display_df_info("Raw Input", input_data)

    # Separate out the outcome variable from the loaded dataframe
    output_var_name = 'SalePrice'
    output_var = input_data[output_var_name]
    input_data.drop(output_var_name, axis=1, inplace=True)

    # DATA ENGINEERING / MODEL DEFINITION
    #####################################

    # Subsetting the columns: define features to keep
    feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd",
                     "HouseStyle"]
    features = input_data[feature_names]
    display_df_info('Features before Transform', features, v=True)

    # Create the pipeline ...
    # 1. Pre-processing
    numerical_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    categorical_features = ['HouseStyle']

    reprocess = make_column_transformer(
        (make_pipeline(GetAge(), SimpleImputer(), StandardScaler()), numerical_features),
        (OneHotEncoder(), categorical_features)
    )

    # 2. Combine pre-processing with ML algorithm
    model = make_pipeline(
        reprocess,  # Use reprocess here instead of preprocess
        LinearRegression()  # You can replace this with any other scikit-learn regression algorithm
    )

    # TRAINING
    ##########
    # Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(features, output_var, test_size=0.3, random_state=42)

    # Train the pipeline
    model.fit(x_train, y_train)

    # SCORING/EVALUATION
    ####################
    pred_test = model.predict(x_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    r2 = r2_score(y_test, pred_test)
    print("Results on Test Data")
    print("####################")
    print("RMSE: {:.2f}".format(rmse))
    print("R2 Score: {:.5f}".format(r2))

    # Compare actual vs predicted values
    compare = pd.DataFrame({
        "Actual": y_test,
        "Prediction": pred_test,
        "Difference": y_test - pred_test
    })
    display_df_info('Actual vs Predicted Comparison', compare)

    # Save the model
    with open('my_model_lr.joblib', 'wb') as fo:
        joblib.dump(model, fo)


if __name__ == '__main__':
    main()
