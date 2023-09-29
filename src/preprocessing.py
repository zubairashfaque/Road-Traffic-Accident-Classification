import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
import os
import sys
import joblib
import copy
def preprocess_data(input_file, output_file, ordinal_encoder_mapping_file):
    """
    Preprocesses a CSV file containing RTA data.

    This function reads a CSV file containing RTA data, performs various data preprocessing steps, and saves the
    processed data to a new CSV file.

    Args:
        input_file (str): The path to the input CSV file containing the raw RTA data.
        output_file (str): The path to the output CSV file where the processed data will be saved.
        ordinal_encoder_mapping_file (str): The path to the JSON file where ordinal encoder mappings will be saved.
        label_encoder_mapping_file (str): The path to the JSON file where label encoder mappings will be saved.

    Returns:
        None
    """

    # Step 1: Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    print("Step 1: CSV file read into DataFrame.")

    # Step 2: Drop unnecessary columns
    df.drop(['Pedestrian_movement', 'Road_allignment', 'Fitness_of_casuality'], axis=1, inplace=True)
    print("Step 2: Unnecessary columns dropped.")

    # Define the target column
    target = 'Accident_severity'

    # Step 3: Split the data into folds using StratifiedKFold
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df[target].values)):
        df.loc[val_idx, "kfold"] = fold
    print("Step 3: Split the data into folds using StratifiedKFold")

    # Step 4: Convert the 'Time' column to datetime and extract hour and minute
    df['Time'] = df['Time'].astype('datetime64[ns]')
    df['hour'] = df['Time'].dt.hour
    df['minute'] = df['Time'].dt.minute
    print("Step 4: 'Time' column converted to datetime.")

    # Define a function to categorize the 'hour' into sessions
    def divide_day(x):
        if (x > 4) and (x <= 8):
            return 'Early Morning'
        elif (x > 8) and (x <= 12):
            return 'Morning'
        elif (x > 12) and (x <= 16):
            return 'Noon'
        elif (x > 16) and (x <= 20):
            return 'Evening'
        elif (x > 20) and (x <= 24):
            return 'Night'
        elif x <= 4:
            return 'Late Night'

    # Step 5: Apply the 'divide_day' function to create a 'session' column
    df['session'] = df['hour'].apply(divide_day)
    print("Step 5: 'session' column created based on 'hour.")

    # Step 6: Drop the original 'Time' column
    df = df.drop("Time", axis=1)
    df_deep_copy = copy.deepcopy(df)
    # Step 7: Perform ordinal encoding on categorical columns (excluding target)
    encoder = OrdinalEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
    ordinal_encoders = {}  # Dictionary to store ordinal encoder mappings

    joblib.dump(encoder, 'data/processed/ordinal_encoder.pkl')

    # Save the ordinal encoder for specific columns to a separate file
    columns_to_encode = ['Age_band_of_driver', 'Light_conditions', 'Day_of_week', 'Types_of_Junction', 'Lanes_or_Medians', 'session']
    df_deep_copy[columns_to_encode] = encoder.fit_transform(df_deep_copy[columns_to_encode])
    joblib.dump(encoder, 'data/processed/ordinal_encoder_final.pkl')

    # Step 13: Define the number of neighbors for KNN imputation
    k_neighbors = 5

    # Initialize the KNN imputer
    imputer = KNNImputer(n_neighbors=k_neighbors)

    # Step 9: Perform KNN imputation on the dataset
    imputed_data = imputer.fit_transform(df)
    print("Step 10: Missing values imputed using KNNImputer.")

    # Convert the imputed data back to a DataFrame
    df = pd.DataFrame(imputed_data, columns=df.columns)

    # Step 10: Save the processed DataFrame to the output CSV file
    df.to_csv(output_file, index=False)
    print(f"Step 11: Processed data saved to {output_file}.")

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython cross_validation.py data-dir-path output-dir-path\n")
        sys.exit(1)

    train_input_cv = os.path.join(sys.argv[1], "RTA_Dataset.csv")
    train_output_cv = os.path.join(sys.argv[2], "FE_output.csv")

    ordinal_encoder_mapping_file = os.path.join(sys.argv[2], "ordinal_encoder_mapping.json")


    preprocess_data(train_input_cv, train_output_cv, ordinal_encoder_mapping_file)
