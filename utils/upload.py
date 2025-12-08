# download files from huggingface to a local directory
import os
import shutil
import tarfile
import tempfile
import time
from huggingface_hub import hf_hub_download, HfApi, login
from tqdm import tqdm
            
            
def tar_batch_and_push_to_huggingface(local_directory, repo_name, type_ = 'images', skip = -1):
    # Login to Hugging Face
    # Initialize Hugging Face API
    api = HfApi()

    # Get all image files from the directory
    image_files = sorted([f for f in os.listdir(local_directory) if os.path.isdir(os.path.join(local_directory, f))])
    
    # Process and upload in batches
    if type_ == 'folder':
        for idx, folder_name in enumerate(image_files):
            if skip >= 0 and idx < skip:
                continue
                
            with tempfile.TemporaryDirectory() as temp_dir:
                tar_filename = f"{folder_name}.tar.gz"
                tar_path = os.path.join(temp_dir, tar_filename)
                
                print(f"Creating tar file for folder {folder_name}: {tar_path}")
                folder_path = os.path.join(local_directory, folder_name)
                all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                
                with tarfile.open(tar_path, "w:gz") as tar:
                    for file in tqdm(all_files, desc=f"Adding files to {tar_filename}"):
                        file_path = os.path.join(folder_path, file)
                        tar.add(file_path, arcname=file)
                
                print(f"Uploading {tar_filename} to Hugging Face...")

                # Upload tar file to Hugging Face
                api.upload_file(
                    path_or_fileobj=tar_path,
                    path_in_repo=tar_filename,
                    repo_id=repo_name,
                    repo_type="dataset",
                )

                print(f"Upload complete. Tar file available at: https://huggingface.co/datasets/{repo_name}/resolve/main/{tar_filename}")
    print("All batches have been uploaded to Hugging Face.")
    

    
# Need another solution for downloading large files from huggingface

# Usage example
if __name__ == '__main__':
    repo_name = "hung20gg/cc12m_siglip_b224"
    local_directory = "../../data/cc12m-siglip-b224/numpy"

    tar_batch_and_push_to_huggingface(local_directory=local_directory, repo_name=repo_name, type_='folder', skip=25)