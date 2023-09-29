from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
import sys
import os
import pandas as pd
import warnings

# Create SMOTE (oversampler) with specific configurations
sm = SMOTE(
    sampling_strategy='auto',  # samples only the minority class
    random_state=0,  # for reproducibility
    k_neighbors=5,
    n_jobs=4
)

# Suppress the specific warning related to the n_jobs parameter in SMOTE
# Suppress the specific warning related to the n_jobs parameter in SMOTE
warnings.filterwarnings("ignore", category=FutureWarning, module="imblearn.over_sampling._smote.base")

# Create EditedNearestNeighbours (undersampler) with specific configurations
enn = EditedNearestNeighbours(
    sampling_strategy='auto',
    n_neighbors=3,
    kind_sel='all',
    n_jobs=4
)

# Create TomekLinks (undersampler) with specific configurations
tl = TomekLinks(
    sampling_strategy='all',
    n_jobs=4
)

# Define a dictionary for undersampled-oversampled datasets
under_oversamp_dict = {

    'under_oversamp_smtomek': SMOTETomek(
        sampling_strategy='auto',  # samples only the minority class
        random_state=0,  # for reproducibility
        smote=sm,  # Use the previously defined SMOTE oversampler
        tomek=tl,  # Use TomekLinks as an undersampler
        n_jobs=4
    ),

    'under_oversamp_smenn': SMOTEENN(
        sampling_strategy='auto',  # samples only the minority class
        random_state=0,  # for reproducibility
        smote=sm,  # Use the previously defined SMOTE oversampler
        enn=enn,  # Use EditedNearestNeighbours as an undersampler
        n_jobs=4
    )
}

def under_oversamp_run(fold, target="Accident_severity", input_dir="data/processed", output_dir="data/processed/sample_data"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the processed data from the input file
    input_file = os.path.join(input_dir, "FE_output.csv")
    df = pd.read_csv(input_file)

    # Filter training data where kfold is not equal to the provided fold
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # Filter validation data where kfold is equal to the provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Extract the label column and the features
    y = df_train[target]
    X = df_train.drop([target, "kfold"], axis=1)

    # Loop through each oversampler and generate synthetic samples
    for under_oversamp_name, under_oversamp in under_oversamp_dict.items():
        X_resampled, y_resampled = under_oversamp.fit_resample(X, y)
        combined_df_train = pd.concat([X_resampled, y_resampled], axis=1)



        # Define file names for the resampled train and test data
        train_output_file = os.path.join(output_dir, f"{under_oversamp_name}_{fold}_train.csv")
        test_output_file = os.path.join(output_dir, f"{under_oversamp_name}_{fold}_test.csv")

        # Save the resampled train data to a CSV file
        combined_df_train.to_csv(train_output_file, index=False)

        # Save the validation data (test data) to a CSV file
        df_valid.to_csv(test_output_file, index=False)

        # Remove the "oversampler_" prefix from the oversampler name for better clarity
        under_oversamp_name = under_oversamp_name.replace("under_oversamp_", "")
        # Print a message indicating the oversampler technique used and the fold processed
        print(f"Generates synthetic samples using {under_oversamp_name} for fold {fold}.")


# Example usage:
if __name__ == "__main__":
    # Check for correct command-line arguments
    if len(sys.argv) != 3 and len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython cross_validation.py data-dir-path output-dir-path\n")
        sys.exit(1)

    # Get input and output directory paths from command-line arguments
    input_d = sys.argv[1]
    output_d = sys.argv[2]

    # Loop through folds (assuming 5-fold cross-validation) and apply oversampling
    for fold_ in range(5):
        under_oversamp_run(fold_, input_dir=input_d, output_dir=output_d)