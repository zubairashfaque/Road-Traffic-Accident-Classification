import os
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from sklearn.svm import SVC
import sys
import warnings

# Suppress the specific warning
warnings.filterwarnings("ignore", category=FutureWarning, module="imblearn.over_sampling._smote.filter")

# Suppress the specific warning related to n_jobs in ADASYN
warnings.filterwarnings("ignore", category=FutureWarning, module="imblearn.over_sampling._adasyn")

# Suppress the specific warning related to n_jobs in imblearn.over_sampling
warnings.filterwarnings("ignore", category=FutureWarning, module="imblearn.over_sampling")


oversampler_dict = {

    'oversampler_random': RandomOverSampler(
        sampling_strategy='auto',
        random_state=0),

    'oversampler_smote': SMOTE(
        sampling_strategy='auto',  # samples only the minority class
        random_state=0,  # for reproducibility
        k_neighbors=5,
        n_jobs=4),

    'oversampler_adasyn': ADASYN(
        sampling_strategy='auto',  # samples only the minority class
        random_state=0,  # for reproducibility
        n_neighbors=5,
        n_jobs=4),

    'oversampler_border1': BorderlineSMOTE(
        sampling_strategy='auto',  # samples only the minority class
        random_state=0,  # for reproducibility
        k_neighbors=5,
        m_neighbors=10,
        kind='borderline-1',
        n_jobs=4),

    'oversampler_border2': BorderlineSMOTE(
        sampling_strategy='auto',  # samples only the minority class
        random_state=0,  # for reproducibility
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


def oversampler_run(fold, target="Accident_severity", input_dir="data/processed", output_dir="data/processed/sample_data"):
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

    for oversampler_name, oversampler in oversampler_dict.items():
        X_resampled, y_resampled = oversampler.fit_resample(X, y)
        combined_df_train = pd.concat([X_resampled, y_resampled], axis=1)

        # Define file names for train and test data
        train_output_file = os.path.join(output_dir, f"{oversampler_name}_{fold}_train.csv")
        test_output_file = os.path.join(output_dir, f"{oversampler_name}_{fold}_test.csv")

        # Save the resampled train data
        combined_df_train.to_csv(train_output_file, index=False)

        # Save the validation data as the test data
        df_valid.to_csv(test_output_file, index=False)

        oversampler_name = oversampler_name.replace("oversampler_", "")

        # Print the oversampler type and fold after saving
        print(f"Generates synthetic samples using technique {oversampler_name} data for fold {fold}.")



# Example usage:
if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython cross_validation.py data-dir-path output-dir-path\n")
        sys.exit(1)

    input_d = sys.argv[1]
    output_d = sys.argv[2]

    for fold_ in range(5):
        oversampler_run(fold_, input_dir=input_d, output_dir=output_d)