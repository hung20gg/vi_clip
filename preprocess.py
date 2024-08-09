# download files from huggingface to a local directory
import os
import tarfile
import tempfile
from huggingface_hub import hf_hub_download, HfApi, login
from tqdm import tqdm

def download_and_extract_batches(repo_name, local_directory):
    # Initialize Hugging Face API
    api = HfApi()

    # Create the local directory if it doesn't exist
    os.makedirs(local_directory, exist_ok=True)

    # Get the list of files in the repository
    files = api.list_repo_files(repo_id=repo_name, repo_type="dataset")

    # Filter for tar.gz files
    tar_files = [f for f in files if f.endswith('.tar.gz')]

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
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=local_directory)

        print(f"Extracted {tar_file} to {local_directory}")

    print("All batches have been downloaded and extracted.")

# Usage example
if __name__ == '__main__':
    repo_name = "hung20gg/your-dataset-name"
    local_directory = "../data/sample"

    download_and_extract_batches(repo_name, local_directory)