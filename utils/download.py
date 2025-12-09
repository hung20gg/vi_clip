# download files from huggingface to a local directory
import os
import shutil
import tarfile
import tempfile
import time
from huggingface_hub import hf_hub_download, HfApi, login
from tqdm import tqdm

def delete_cache_files(local_directory, repo_name):
    lock_file = os.path.join(local_directory, ".locks")
    dataset_cache = os.path.join(local_directory, f"datasets--{repo_name.replace('/', '--')}")
    
    if os.path.exists(lock_file):
        try:
            shutil.rmtree(lock_file)
            print(f"Deleted {lock_file}")
        except:
            print(f"Could not delete {lock_file}")
        
    if os.path.exists(dataset_cache):
        try:
            shutil.rmtree(dataset_cache)
            print(f"Deleted {dataset_cache}")
        except:
            print(f"Could not delete {dataset_cache}")

def download_and_extract_batches(repo_name, local_directory, folder_name = "images", keep_folder = False, ending='.tar.gz'):
    # Initialize Hugging Face API
    start = time.time()
    api = HfApi()
    end = time.time()
    print(f"Initialized Hugging Face API in {end - start:.2f} seconds.")

    # Create the local directory if it doesn't exist
    os.makedirs(local_directory, exist_ok=True)
    images_directory = os.path.join(local_directory, folder_name)
    os.makedirs(images_directory, exist_ok=True)
    print(images_directory)

    # Get the list of files in the repository
    files = api.list_repo_files(repo_id=repo_name, repo_type="dataset")
    # Filter for tar.gz files
    
    print(f"Found {len(files)} files in the repository.")
    print(files)
    
    tar_files = [f for f in files if f.endswith(ending)]
    parquet_files = [f for f in files if f.endswith('.parquet')]
    
    # Something is wrong while downloading parquet files (can not save to local directory)
    for parquet_file in tqdm(parquet_files, desc="Downloading parquet files"):
        # Download the parquet file
        parquet_path = hf_hub_download(
            repo_id=repo_name,
            filename=parquet_file,
            repo_type="dataset",
            cache_dir=local_directory
        )
        
        print(f"Downloaded {parquet_file} to {local_directory}")
        
        # Test copy
        shutil.copy(parquet_path, os.path.join(local_directory, 'captions.parquet'))
        
        print(f"Copied {parquet_file} to {local_directory}")

    for tar_file in tqdm(tar_files, desc="Downloading and extracting batches"):
        # Create a temporary directory for this batch
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download the tar file
            tar_path = hf_hub_download(
                repo_id=repo_name,
                filename=tar_file,
                repo_type="dataset",
                cache_dir=temp_dir
            )

            # Extract the tar file
            if keep_folder:
                folder_name = tar_file.split('.')[0]
                images_folder = os.path.join(images_directory, folder_name)
                os.makedirs(images_folder, exist_ok=True)
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=images_folder)
            else:
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=images_directory)

        print(f"Extracted {tar_file} to {images_directory}")

    print("All batches have been downloaded and extracted.")
    
    delete_cache_files(local_directory, repo_name)
    

    
# Need another solution for downloading large files from huggingface

# Usage example
if __name__ == '__main__':
    repo_name = "hung20gg/2M5_cc3m_clip_B224"
    local_directory = "../../data/2M5_cc3m_clip_B224"
    folder_name = "numpy"

    download_and_extract_batches(repo_name, local_directory, folder_name=folder_name, keep_folder=True)

    repo_name = "hung20gg/cc12m_clip_b224"
    local_directory = "../../data/cc12m-clip-b224"
    folder_name = "numpy"

    download_and_extract_batches(repo_name, local_directory, folder_name=folder_name, keep_folder=True)
