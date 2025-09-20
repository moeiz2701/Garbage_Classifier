# Garbage Classification Dataset Downloader
# Dataset: https://www.kaggle.com/datasets/mostafaabla/garbage-classification

import os
import zipfile


# Step 1: Install required dependencies
def install_dependencies():
    import subprocess
    subprocess.run(['pip', 'install', 'kaggle', '--upgrade'])
    subprocess.run(['pip', 'install', 'scikit-learn', 'scikit-image', 'Pillow', 'matplotlib'])

# Step 2: Set up Kaggle API authentication
def setup_kaggle_auth():
    # Method 1: Try using environment variables
    os.environ['KAGGLE_USERNAME'] = 'moiz2701'
    os.environ['KAGGLE_KEY'] = '0e29c796624f4607337086263f4ad32b'
    
    # Method 2: Create kaggle.json file if it doesn't exist
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_json_path):
        print("Creating kaggle.json file...")
        os.makedirs(kaggle_dir, exist_ok=True)
        
        kaggle_config = {
            "username": "moiz2701",
            "key": "0e29c796624f4607337086263f4ad32b"
        }
        
        import json
        with open(kaggle_json_path, 'w') as f:
            json.dump(kaggle_config, f)
        
        # Set proper permissions (important for security)
        os.chmod(kaggle_json_path, 0o600)
        print(f"Created {kaggle_json_path}")
    
    # Import and create API instance after setting up credentials
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    
    try:
        api.authenticate()
        print("Kaggle API authentication successful")
        return api
    except Exception as e:
        print(f"Kaggle API authentication failed: {e}")
        print("Please check your Kaggle credentials or dataset access permissions")
        raise

# Step 3: Download and extract dataset
def download_dataset(api, dataset_name="mostafaabla/garbage-classification"):
    download_path = "dataset"
    os.makedirs(download_path, exist_ok=True)

    try:
        api.dataset_download_files(dataset_name, path=download_path, unzip=False)
        print("Dataset downloaded successfully")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have accepted the dataset terms at https://www.kaggle.com/datasets/mostafaabla/garbage-classification")
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
    data_dir = download_dataset(api)
    
    # Define class names
   
    
    print("\nDataset setup completed successfully!")

if __name__ == "__main__":
    main() 