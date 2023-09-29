import os
import pandas as pd
import warnings
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, OneSidedSelection, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, NearMiss
import sys


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

def undersampler_run(fold, target="Accident_severity", input_dir="data/processed", output_dir="data/processed/sample_data"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the processed data from the input file
    input_file = os.path.join(input_dir, "FE_output.csv")
    df = pd.read_csv(input_file)

    # training data is where kfold is not equal to provided fold
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column from dataframe and convert it to
    y = df_train[target]
    X = df_train.drop([target, "kfold"], axis=1)

    for undersampler_name, undersampler in undersampler_dict.items():
        X_resampled, y_resampled = undersampler.fit_resample(X, y)
        combined_df_train = pd.concat([X_resampled, y_resampled], axis=1)

        # Define file names for train and test data
        train_output_file = os.path.join(output_dir, f"{undersampler_name}_{fold}_train.csv")
        test_output_file = os.path.join(output_dir, f"{undersampler_name}_{fold}_test.csv")

        # Save the resampled train data
        combined_df_train.to_csv(train_output_file, index=False)

        # Save the validation data as the test data
        df_valid.to_csv(test_output_file, index=False)

        # Print the undersampler type (without "under_") and fold after saving
        #print(f"Saved {undersampler_name} data for fold {fold}.")

        undersampler_name = undersampler_name.replace("under_", "")

        # Print the oversampler type and fold after saving
        print(f"Generates synthetic samples using technique {undersampler_name} data for fold {fold}.")

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython cross_validation.py data-dir-path output-dir-path\n")
        sys.exit(1)

    input_d = sys.argv[1]
    output_d = sys.argv[2]

    for fold_ in range(5):
        undersampler_run(fold_, input_dir=input_d, output_dir=output_d)
