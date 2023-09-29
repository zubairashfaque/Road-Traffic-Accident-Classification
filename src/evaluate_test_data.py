import os
import yaml
import pandas as pd
from sklearn.metrics import f1_score
from joblib import load
from collections import OrderedDict
from sklearn.ensemble import ExtraTreesClassifier

# Step 1: Load data from the YAML file
with open("data/final/top_f1_score.yaml", "r") as yaml_file:
    data = yaml.safe_load(yaml_file)

# Step 2: Extract the list of file names
file_names = data['top_f1_score']

# Step 3: Define the directory where feature importances were saved
feature_importance_dir = "data/final/feature_importances"

# Step 4: Define the directory where test data is located
test_data_dir = "data/processed/sample_data"

# Step 5: Initialize an empty list to store results
test_results = []

# Step 6: Loop through each train file (feature importance file) and find the corresponding test file
for train_file in file_names:
    # Step 7: Load the corresponding top model result for this train file
    top_model_results_file = os.path.join(feature_importance_dir, 'top_model_results.csv')
    top_model_results_df = pd.read_csv(top_model_results_file)

    # Step 8: Find the row in the DataFrame corresponding to the train file
    row = top_model_results_df[top_model_results_df['Filename'] == train_file]

    # Step 9: Extract the hyperparameters as an ordered dictionary
    hyperparameters = OrderedDict(eval(row['Best Model Hyperparameters'].values[0]))

    # Step 10: Determine the corresponding test file
    test_file = train_file.replace("_train.csv", "_test.csv")
    test_data_file = os.path.join(test_data_dir, test_file)

    # Step 11: Load the test data
    test_data = pd.read_csv(test_data_file)

    # Step 12: Split the test data into features (X_test) and target (y_test)
    X_test = test_data.drop(columns=['Accident_severity', 'kfold'])
    y_test = test_data['Accident_severity']

    # Step 13: Load the best model for this train file (ExtraTreesClassifier)
    et_classifier = ExtraTreesClassifier(max_features=None)

    # Step 14: Fit the classifier with the training data
    X_train = pd.read_csv(os.path.join('data/processed/sample_data', train_file))
    X_train = X_train.drop(columns=['Accident_severity'])
    y_train = pd.read_csv(os.path.join('data/processed/sample_data', train_file))['Accident_severity']
    et_classifier.set_params(**hyperparameters)
    et_classifier.fit(X_train, y_train)

    # Step 15: Make predictions on the test data using the loaded hyperparameters
    y_pred = et_classifier.predict(X_test)

    # Step 16: Calculate the F1 weighted score for the predictions
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # Step 17: Append the results to the list
    test_results.append({
        'Filename': test_file,
        'Best Model Hyperparameters': hyperparameters,
        'F1 Weighted Score': f1_weighted
    })

# Step 18: Create a DataFrame from the test results list
test_results_df = pd.DataFrame(test_results)

# Step 19: Save the test results DataFrame to a CSV file
test_results_file_path = os.path.join(feature_importance_dir, 'test_results.csv')
test_results_df.to_csv(test_results_file_path, index=False)

# Step 20: Print a message indicating the script has completed execution
print("Script execution completed.")