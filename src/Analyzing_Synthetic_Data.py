import os
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE)
from imblearn.under_sampling import (RandomUnderSampler, TomekLinks, OneSidedSelection, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, NearMiss)
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import yaml

# Define the undersampler dictionary
undersampler_dict = {
    'under_random': RandomUnderSampler(
        sampling_strategy='auto',
        random_state=0,
        replacement=False),

    'under_tomek': TomekLinks(
        sampling_strategy='auto',
        n_jobs=4),

    'under_oss': OneSidedSelection(
        sampling_strategy='auto',
        random_state=0,
        n_neighbors=1,
        n_jobs=4),

    'under_enn': EditedNearestNeighbours(
        sampling_strategy='auto',
        n_neighbors=3,
        kind_sel='all',
        n_jobs=4),

    'under_renn': RepeatedEditedNearestNeighbours(
        sampling_strategy='auto',
        n_neighbors=3,
        kind_sel='all',
        n_jobs=4,
        max_iter=100),

    'under_allknn': AllKNN(
        sampling_strategy='auto',
        n_neighbors=5,
        kind_sel='all',
        n_jobs=4),

    'under_nm1': NearMiss(
        sampling_strategy='auto',
        version=1,
        n_neighbors=3,
        n_jobs=4),

    'under_nm2': NearMiss(
        sampling_strategy='auto',
        version=2,
        n_neighbors=3,
        n_jobs=4)
}

# Define the oversampler dictionary
oversampler_dict = {
    'oversampler_random': RandomOverSampler(
        sampling_strategy='auto',
        random_state=0),

    'oversampler_smote': SMOTE(
        sampling_strategy='auto',
        random_state=0,
        k_neighbors=5,
        n_jobs=4),

    'oversampler_adasyn': ADASYN(
        sampling_strategy='auto',
        random_state=0,
        n_neighbors=5,
        n_jobs=4),

    'oversampler_border1': BorderlineSMOTE(
        sampling_strategy='auto',
        random_state=0,
        k_neighbors=5,
        m_neighbors=10,
        kind='borderline-1',
        n_jobs=4),

    'oversampler_border2': BorderlineSMOTE(
        sampling_strategy='auto',
        random_state=0,
        k_neighbors=5,
        m_neighbors=10,
        kind='borderline-2',
        n_jobs=4),

    'oversampler_svm': SVMSMOTE(
        sampling_strategy='auto',  # samples only the minority class
        random_state=0,  # for reproducibility
        k_neighbors=5,
        m_neighbors=10,
        n_jobs=4,
        svm_estimator=SVC(kernel='linear')),
}

# Define the combined sampler dictionary
under_oversamp_dict = {
    'under_oversamp_smtomek': SMOTETomek(
        sampling_strategy='auto',
        random_state=0,
        smote=SMOTE(sampling_strategy='auto', random_state=0, k_neighbors=5, n_jobs=4),
        tomek=TomekLinks(sampling_strategy='all', n_jobs=4),
        n_jobs=4
    ),

    'under_oversamp_smenn': SMOTEENN(
        sampling_strategy='auto',
        random_state=0,
        smote=SMOTE(sampling_strategy='auto', random_state=0, k_neighbors=5, n_jobs=4),
        enn=EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=3, kind_sel='all', n_jobs=4),
        n_jobs=4
    ),
}
# Define a dictionary with folds and methods
param_dict = {
    "folds": [0],  # Add more folds if needed
#   "folds": [0, 1, 2, 3, 4],  # Add more folds if needed
    "methods": {
        "oversampling": oversampler_dict,
        "undersampling": undersampler_dict,
        "under_oversampling": under_oversamp_dict,
    },
}

# Function to train and evaluate a classifier using sampling
def sampler_run_2(fold, sampler_dict, target="Accident_severity", output_dir="data/processed/sample_data"):
    results = []

    for sampler_type, sampler_methods in sampler_dict.items():
        for sampler_name, sampler in sampler_methods.items():

            print(f"Processing fold {fold}, sampler type: {sampler_type}, sampler name: {sampler_name}")

            # Load your existing code for oversampling or undersampling here
            file_name_train = f"{sampler_name}_" + str(fold) + '_' + 'train' + '.csv'
            train_output_cv = os.path.join(output_dir, file_name_train)
            file_name_test = f"{sampler_name}_" + str(fold) + '_' + 'test' + '.csv'
            test_output_cv = os.path.join(output_dir, file_name_test)
            df_train = pd.read_csv(train_output_cv)
            df_valid = pd.read_csv(test_output_cv)
            y_train = df_train[target].values
            df_train = df_train.drop([target], axis=1)
            X_train = df_train.values
            y_valid = df_valid[target].values
            df_valid = df_valid.drop([target, "kfold"], axis=1)
            X_valid = df_valid.values

            # Create a decision tree classifier
            clf = DecisionTreeClassifier(random_state=42)
            clf = RandomForestClassifier(random_state=42)

            # Fit the model on the training data for this fold
            clf.fit(X_train, y_train)

            y_pred_valid = clf.predict(X_valid)
            f1 = f1_score(y_valid, y_pred_valid, average='weighted')

            results.append({
                "Fold": fold,
                "Method": f"{sampler_type}_{sampler_name}",
                "F1_Score": f1,
                "File_name": file_name_test
            })

    return results

# Main script to generate the graph
# Main script to generate the graph
def generate_f1_score_graph():
    all_results = []

    for fold in param_dict["folds"]:
        sampler_results = sampler_run_2(fold, param_dict["methods"], target="Accident_severity", output_dir="data/processed/sample_data")
        all_results.extend(sampler_results)

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(by='F1_Score', ascending=False)

    # Create a Plotly bar graph
    fig = px.bar(
        results_df,
        x="File_name",
        y="F1_Score",
        color="Method",
        title="F1 Score Comparison for Different Folds and Sampling Methods",
        labels={"F1_Score": "F1 Score"}
    )

    # Change the output directory here
    output = "notebooks/Analyzing_Synthetic_Data"

    # Create the output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)

    # Save the graph as an HTML file in the output directory
    fig.write_html(os.path.join(output, "f1_score_comparison.html"))

    # Assuming 'df' is your DataFrame containing the data
    average_scores = results_df.groupby('Method')['F1_Score'].mean().reset_index()

    # Sort the DataFrame by average F1 Score in descending order
    average_scores = average_scores.sort_values(by='F1_Score', ascending=False)


    return results_df, average_scores

if __name__ == "__main__":

    result, average_scores = generate_f1_score_graph()


    # Extract the top-average File_names
    top_average_method = average_scores.iloc[0]['Method']
    top_average_file_names = result[result['Method'] == top_average_method]['File_name'].tolist()


    # Extract the top 5 files based on F1 Score
    top_files = result.sort_values(by='F1_Score', ascending=False).head(5)

    # Print the list of top 5 File_names
    top_5_file_names = top_files['File_name'].tolist()

    # Define the path to save the YAML file
    yaml_file_path = "data/processed/sample_data/top_sample_list.yaml"

    # Write the list to the YAML file
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(top_5_file_names, yaml_file, default_flow_style=False)

    print("Print the list of top-average File_names")
    print(top_5_file_names)

    print("top_sample_list.yaml has been created.")