import os
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil

# Set your Kaggle API credentials (replace with your own values)
os.environ["KAGGLE_USERNAME"] = "usrname"
os.environ["KAGGLE_KEY"] = "kaggle_key"

# Specify the dataset you want to download (replace with the dataset you need)
dataset_name = "saurabhshahane/road-traffic-accidents"

# Set the directory where you want to save the downloaded dataset
data_dir = "data/raw"

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Create the data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Download the dataset using the Kaggle API
api.dataset_download_files(dataset_name, path=data_dir, unzip=True)

# List the downloaded files
downloaded_files = os.listdir(data_dir)

# Print the list of downloaded files
print("Downloaded files:")
for file in downloaded_files:
    print(os.path.join(data_dir, file))

# Specify the file to delete and the new filename
file_to_delete = os.path.join(data_dir, "cleaned.csv")
new_file_name = os.path.join(data_dir, "RTA_Dataset.csv")

# Delete the file if it exists
if os.path.exists(file_to_delete):
    os.remove(file_to_delete)
    print(f"Deleted file: {file_to_delete}")

# Rename "RTA Dataset.csv" to "RTA_Dataset.csv" using shutil
if os.path.exists(os.path.join(data_dir, "RTA Dataset.csv")):
    shutil.move(os.path.join(data_dir, "RTA Dataset.csv"), new_file_name)
    print(f"Renamed file: {new_file_name}")
