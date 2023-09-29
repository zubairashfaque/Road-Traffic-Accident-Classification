import os
import yaml
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
# Define the hyperparameter search space
param_space = {
    'n_estimators': Integer(120, 1200, name='n_estimators'),
    'max_depth': Categorical([5, 8, 15, 25, 30, None]),
    'min_samples_split': Integer(2, 15),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Categorical(['sqrt', 'log2']),
    'bootstrap': [True, False]
}

# Create a BayesSearchCV object for ExtraTreesClassifier hyperparameter tuning
et_classifier = ExtraTreesClassifier(max_features=None)
# Create a BayesSearchCV object
bayes_search = BayesSearchCV(
    et_classifier,
    param_space,
    n_iter=50,  # Number of iterations (adjust as needed)
    scoring='f1_weighted',  # Scoring metric for optimization
    cv=5,  # Number of cross-validation folds
    n_jobs=-1,  # Use all available CPU cores
    verbose=1,  # Enable verbose output
    refit=True,  # Refit the best model on the entire dataset
    random_state=42,  # Set a random seed for reproducibility
)

# Load data from the YAML file
with open("data/final/top_f1_score.yaml", "r") as yaml_file:
    data = yaml.safe_load(yaml_file)

# Extract the list of file names
file_names = data['top_f1_score']

# Directory to save feature importances
feature_importance_dir = "data/final/feature_importances"

# Initialize an empty list to store top models and their results
top_model_results = []

# Loop through each train file (feature importance file) and find the top model
for train_file in file_names:
    print(f"Processing file: {train_file}")
    new_filename = train_file.replace("_train.csv", "_Extra Trees_top_10_features.csv")
    file_path = os.path.join(feature_importance_dir, new_filename)
    feature_importance_df = pd.read_csv(file_path)
    print("Top 10 features:", feature_importance_df['Feature'].tolist())

    # Load the corresponding train data file
    train_data_file = os.path.join('data/processed/sample_data', train_file)
    train_data = pd.read_csv(train_data_file)

    # Split the data into features (X) and target (y)
    X_train = train_data.drop(columns=['Accident_severity'])
    y_train = train_data['Accident_severity']

    # Select only the top 10 features based on feature importance
    selected_features = feature_importance_df['Feature'].tolist()
    X_train = X_train[selected_features]

    # Perform Bayesian Optimization on the training data
    bayes_search.fit(X_train, y_train)

    # Get the best hyperparameters and the best score
    best_params = bayes_search.best_params_
    best_score = bayes_search.best_score_

    # Train an ExtraTreesClassifier with the best hyperparameters
    et_classifier.set_params(**best_params)
    et_classifier.fit(X_train, y_train)

    # Append the top model and its results to the list
    top_model_results.append({
        'Filename': train_file,
        'Model_name': 'ExtraTreesClassifier',
        'Best Model Hyperparameters': best_params,
        'Best Model Score (F1 Weighted)': best_score
    })

# Create a DataFrame from the top model results list
top_model_results_df = pd.DataFrame(top_model_results)

# Save the top model results DataFrame to a CSV file
top_model_results_file_path = os.path.join(feature_importance_dir, 'top_model_results.csv')
top_model_results_df.to_csv(top_model_results_file_path, index=False)

# Print a message indicating the script has completed execution
print("Script execution completed.")