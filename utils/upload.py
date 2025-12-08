# download files from huggingface to a local directory
import os
import shutil
import tarfile
import tempfile
import time
from huggingface_hub import hf_hub_download, HfApi, login
from tqdm import tqdm
            
            
def tar_batch_and_push_to_huggingface(local_directory, repo_name, batch_size=25000, type_ = 'images', skip = -1):
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
        
        if batch_number <= skip:
            continue
        
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
    

    
# Need another solution for downloading large files from huggingface

# Usage example
if __name__ == '__main__':
    repo_name = "hung20gg/2M5_cc3m_clip_B224"
    local_directory = "../data/2M5_cc3m_clip_B224"
    folder_name = "numpy"

    tar_batch_and_push_to_huggingface()