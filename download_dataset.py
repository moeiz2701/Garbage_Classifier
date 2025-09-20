# Garbage Classification Dataset Downloader
# Dataset: https://www.kaggle.com/datasets/mostafaabla/garbage-classification

import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


# Step 1: Install required dependencies
def install_dependencies():
    import subprocess
    subprocess.run(['pip', 'install', 'kaggle', '--upgrade'])
    subprocess.run(['pip', 'install', 'scikit-learn', 'scikit-image',
                    'Pillow', 'matplotlib'])


# Step 2: Set up Kaggle API authentication
def setup_kaggle_auth():
    os.environ['KAGGLE_USERNAME'] = 'moiz2701'
    os.environ['KAGGLE_KEY'] = '0e29c796624f4607337086263f4ad32b'

    api = KaggleApi()
    try:
        api.authenticate()
        print("Kaggle API authentication successful")
        return api
    except Exception as e:
        print(f"Kaggle API authentication failed: {e}")
        print("Ensure the provided username and key are valid.")
        raise


# Step 3: Download and extract dataset
def download_dataset(api,
                     dataset_name="mostafaabla/garbage-classification"):
    download_path = "dataset"
    os.makedirs(download_path, exist_ok=True)

    try:
        api.dataset_download_files(dataset_name, path=download_path,
                                   unzip=False)
        print("Dataset downloaded successfully")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have accepted the dataset terms at "
              "https://www.kaggle.com/datasets/mostafaabla/"
              "garbage-classification")
        raise

    # Extract the downloaded zip file
    zip_path = os.path.join(download_path, "garbage-classification.zip")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        print("Dataset extracted successfully")
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        raise

    return os.path.join(download_path, "garbage_classification")


def main():
    # Install dependencies
    install_dependencies()

    # Setup Kaggle authentication
    api = setup_kaggle_auth()

    # Download and extract dataset
    download_dataset(api)

    print("\nDataset setup completed successfully!")


if __name__ == "__main__":
    main()