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
            
            
def tar_batch_and_push_to_huggingface(local_directory, repo_name, batch_size=25000, type_ = 'images'):
    # Login to Hugging Face
    # Initialize Hugging Face API
    api = HfApi()

    # Get all image files from the directory
    if type_ == 'images':
        image_files = [f for f in os.listdir(local_directory) if f.lower().endswith(('.png', '.jpg','.jpeg'))]
    elif type_ == 'numpy':
        
        image_files = [f for f in os.listdir(local_directory) if f.lower().endswith('.npy')]
    elif type_ == 'folder':
        batch_size = 1
        image_files = [f for f in os.listdir(local_directory) if os.path.isdir(os.path.join(local_directory, f))]
    # Process and upload in batches
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]
        batch_number = i // batch_size + 1
        
        with tempfile.TemporaryDirectory() as temp_dir:
            if type_ == 'folder':
                tar_filename = f"{batch[0]}.tar.gz"
                tar_path = os.path.join(temp_dir, tar_filename)
                
                print(f"Loading embedding to {tar_path}")
                numpy_files = [f for f in os.listdir(os.path.join(local_directory, batch[0])) if f.lower().endswith('.npy')]
                with tarfile.open(tar_path, "w:gz") as tar:
                    for file in tqdm(numpy_files, desc=f"Adding files to tar (batch {batch_number})"):
                        file_path = os.path.join(local_directory, batch[0], file)
                        tar.add(file_path, arcname=file)
                        
            else:
                # Create a temporary directory for this batch
                tar_filename = f"batch_{batch_number}.tar.gz"
                tar_path = os.path.join(temp_dir, tar_filename)
                
                print(f"Creating tar file for batch {batch_number}: {tar_path}")
                with tarfile.open(tar_path, "w:gz") as tar:
                    for file in tqdm(batch, desc=f"Adding files to tar (batch {batch_number})"):
                        file_path = os.path.join(local_directory, file)
                        tar.add(file_path, arcname=file)

            print(f"Uploading batch {batch_number} to Hugging Face...")

            # Upload tar file to Hugging Face
            api.upload_file(
                path_or_fileobj=tar_path,
                path_in_repo=tar_filename,
                repo_id=repo_name,
                repo_type="dataset",
            )

            print(f"Batch {batch_number} upload complete. Tar file available at: https://huggingface.co/datasets/{repo_name}/resolve/main/{tar_filename}")

    print("All batches have been uploaded to Hugging Face.")
    

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
                images_directory = os.path.join(local_directory, folder_name)
                os.makedirs(images_directory, exist_ok=True)
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=images_directory)
            else:
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=images_directory)

        print(f"Extracted {tar_file} to {images_directory}")

    print("All batches have been downloaded and extracted.")
    
    delete_cache_files(local_directory, repo_name)
    

    
# Need another solution for downloading large files from huggingface

# Usage example
if __name__ == '__main__':
    repo_name = "hung20gg/your-dataset-name"
    local_directory = "../data/sample"

    download_and_extract_batches(repo_name, local_directory)