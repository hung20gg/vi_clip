"""
Extract CC12M dataset from HuggingFace into organized structure with parquet captions.

Structure:
- Input: Download cc12m-train-<number>.tar files from HuggingFace
- Output:
  - images/<number>/<img_id>.<img>
  - captions.parquet with columns: image_id, image, text_id, caption
"""

import os
import shutil
import tarfile
import tempfile
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging
from typing import List, Dict, Optional
from tqdm import tqdm
from huggingface_hub import hf_hub_download, HfApi
from PIL import Image
import io

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CC12MExtractor:
    """Download from HuggingFace and organize CC12M dataset."""
    
    def __init__(self, repo_name: str, output_dir: str, num_threads: int = 4, resize_to: int = 384):
        """
        Initialize the extractor.
        
        Args:
            repo_name: HuggingFace repository name (e.g., "user/cc12m-dataset")
            output_dir: Directory to save extracted images and captions
            num_threads: Number of threads for parallel extraction
            resize_to: Size to resize images to (default: 384x384)
        """
        self.repo_name = repo_name
        self.output_dir = Path(output_dir)
        self.num_threads = num_threads
        self.resize_to = resize_to
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe list for caption data
        self.captions_data = []
        self.data_lock = Lock()
        
        # Initialize HuggingFace API
        self.api = HfApi()
        
    def get_tar_files_from_repo(self, start_num: int = 0, end_num: int = 5) -> List[str]:
        """
        Get list of tar files in the specified range from HuggingFace repo.
        
        Args:
            start_num: Start number (inclusive)
            end_num: End number (inclusive)
            
        Returns:
            List of tar file names available in the repo
        """
        # Get all files from repo
        all_files = self.api.list_repo_files(repo_id=self.repo_name, repo_type="dataset")
        
        # Filter for tar files in the specified range
        tar_files = []
        for num in range(start_num, end_num + 1):
            tar_name = f"cc12m-train-{num:04d}.tar"
            if tar_name in all_files:
                tar_files.append(tar_name)
            else:
                logger.warning(f"Tar file not found in repo: {tar_name}")
        
        return tar_files
    
    def download_and_extract_tar(self, tar_filename: str) -> Dict[str, int]:
        """
        Download a tar file from HuggingFace and extract it.
        
        Args:
            tar_filename: Name of the tar file to download
            
        Returns:
            Dictionary with extraction statistics
        """
        # Get the number from tar filename (e.g., cc12m-train-0001.tar -> 0001)
        tar_number = tar_filename.replace('.tar', '').split('-')[-1]
        
        # Create subdirectory for this tar's images
        tar_images_dir = self.images_dir / tar_number
        tar_images_dir.mkdir(exist_ok=True)
        
        stats = {'images': 0, 'captions': 0, 'errors': 0}
        local_captions = []
        
        try:
            # Download tar file to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Downloading {tar_filename}...")
                tar_path = hf_hub_download(
                    repo_id=self.repo_name,
                    filename=tar_filename,
                    repo_type="dataset",
                    cache_dir=temp_dir
                )
                
                logger.info(f"Extracting {tar_filename}...")
                # Extract and process
                stats_result = self._extract_and_process_tar(tar_path, tar_number, tar_images_dir)
                stats = stats_result['stats']
                local_captions = stats_result['captions']
            
            # Add local captions to global list (thread-safe)
            with self.data_lock:
                self.captions_data.extend(local_captions)
            
            logger.info(f"Completed {tar_filename}: {stats['images']} images, {stats['captions']} captions")
            
        except Exception as e:
            logger.error(f"Error processing {tar_filename}: {e}")
            stats['errors'] += 1
        
        return stats
    
    def _extract_and_process_tar(self, tar_path: str, tar_number: str, tar_images_dir: Path) -> Dict:
        """
        Extract and process a single tar file (internal method).
        
        Args:
            tar_path: Path to the tar file
            tar_number: The 4-digit number identifier
            tar_images_dir: Directory to save images
            
        Returns:
            Dictionary with stats and captions
        """
        stats = {'images': 0, 'captions': 0, 'errors': 0}
        local_captions = []
        
        try:
            with tarfile.open(tar_path, 'r') as tar:
                members = tar.getmembers()
                
                # Group members by img_id
                img_groups = {}
                for member in members:
                    if member.isfile():
                        filename = Path(member.name).name
                        # Extract img_id (first 10 digits before extension)
                        parts = filename.split('.')
                        if len(parts) >= 2:
                            img_id = parts[0]
                            if img_id not in img_groups:
                                img_groups[img_id] = {}
                            
                            ext = parts[-1]
                            if ext == 'txt':
                                img_groups[img_id]['txt'] = member
                            elif ext == 'json':
                                img_groups[img_id]['json'] = member
                            else:
                                # Assume it's an image
                                img_groups[img_id]['img'] = member
                                img_groups[img_id]['img_ext'] = ext
                
                # Process each image group
                for img_id, files in img_groups.items():
                    try:
                        # Extract image
                        if 'img' in files:
                            img_member = files['img']
                            img_ext = files.get('img_ext', 'jpg')
                            
                            # New image filename (save as jpg after resizing)
                            new_img_name = f"{img_id}.jpg"
                            img_output_path = tar_images_dir / new_img_name
                            
                            # Extract, resize and save image
                            with tar.extractfile(img_member) as src:
                                img_bytes = src.read()
                                img = Image.open(io.BytesIO(img_bytes))
                                
                                # Convert to RGB if necessary
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                
                                # Resize to 384x384
                                img_resized = img.resize((self.resize_to, self.resize_to), Image.Resampling.LANCZOS)
                                
                                # Save resized image
                                img_resized.save(img_output_path, 'JPEG', quality=95)
                            
                            stats['images'] += 1
                            
                            # Extract caption from txt file
                            caption = ""
                            if 'txt' in files:
                                txt_member = files['txt']
                                with tar.extractfile(txt_member) as txt_file:
                                    caption = txt_file.read().decode('utf-8', errors='ignore').strip()
                                stats['captions'] += 1
                            
                            # Create caption entry
                            image_id = f"{tar_number}{img_id}"
                            image_name = f"{tar_number}/{new_img_name}"
                            
                            local_captions.append({
                                'image_id': image_id,
                                'image': image_name,
                                'text_id': img_id,
                                'caption': caption
                            })
                    
                    except Exception as e:
                        logger.error(f"Error processing {img_id}: {e}")
                        stats['errors'] += 1
            
        except Exception as e:
            logger.error(f"Error extracting tar: {e}")
            stats['errors'] += 1
        
        return {'stats': stats, 'captions': local_captions}
    
    def download_and_extract_all(self, start_num: int = 0, end_num: int = 5) -> None:
        """
        Download and extract all tar files in the specified range using multithreading.
        
        Args:
            start_num: Start number (inclusive)
            end_num: End number (inclusive)
        """
        # Store range for parquet filename
        self.start_num = start_num
        self.end_num = end_num
        
        logger.info("=" * 60)
        logger.info(f"CC12M Extractor - Repository: {self.repo_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Range: cc12m-train-{start_num:04d}.tar to cc12m-train-{end_num:04d}.tar")
        logger.info("=" * 60)
        
        # Get available tar files from repo
        tar_files = self.get_tar_files_from_repo(start_num, end_num)
        
        if not tar_files:
            logger.error("No tar files found in the repository!")
            return
        
        logger.info(f"Found {len(tar_files)} tar files to process")
        logger.info(f"Using {self.num_threads} threads")
        
        total_stats = {'images': 0, 'captions': 0, 'errors': 0}
        
        # Download and extract tar files in parallel
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {executor.submit(self.download_and_extract_tar, tar_file): tar_file 
                      for tar_file in tar_files}
            
            with tqdm(total=len(tar_files), desc="Processing tar files") as pbar:
                for future in as_completed(futures):
                    stats = future.result()
                    for key in total_stats:
                        total_stats[key] += stats[key]
                    pbar.update(1)
        
        logger.info("=" * 60)
        logger.info(f"Processing complete!")
        logger.info(f"Total: {total_stats['images']} images, "
                   f"{total_stats['captions']} captions, {total_stats['errors']} errors")
        logger.info("=" * 60)
        
        # Save captions to parquet
        self.save_captions_parquet()
        
        # Clean up cache
        self._cleanup_cache()
    
    def save_captions_parquet(self) -> None:
        """Save all captions to a parquet file."""
        if not self.captions_data:
            logger.warning("No captions data to save!")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.captions_data)
        
        # Sort by image_id for consistency
        df = df.sort_values('image_id').reset_index(drop=True)
        
        # Save to parquet with range in filename
        parquet_filename = f"captions_{self.start_num:04d}_{self.end_num:04d}.parquet"
        parquet_path = self.output_dir / parquet_filename
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        
        logger.info("=" * 60)
        logger.info(f"Saved {len(df)} captions to {parquet_path}")
        logger.info(f"Parquet columns: {list(df.columns)}")
        logger.info(f"Sample data:\n{df.head()}")
        logger.info("=" * 60)
    
    def _cleanup_cache(self) -> None:
        """Clean up HuggingFace cache files."""
        cache_dir = self.output_dir / ".cache"
        lock_file = self.output_dir / ".locks"
        dataset_cache = self.output_dir / f"datasets--{self.repo_name.replace('/', '--')}"
        
        for path in [cache_dir, lock_file, dataset_cache]:
            if path.exists():
                try:
                    shutil.rmtree(path)
                    logger.info(f"Cleaned up cache: {path}")
                except Exception as e:
                    logger.warning(f"Could not delete {path}: {e}")


def main():
    """Main execution function."""
    # Configuration
    REPO_NAME = "username/cc12m-dataset"  # Change this to your HuggingFace repo
    OUTPUT_DIR = "/path/to/output"  # Change this to your desired output directory
    START_NUM = 0  # Start number (e.g., 0 for cc12m-train-0000.tar)
    END_NUM = 5    # End number (e.g., 5 for cc12m-train-0005.tar)
    NUM_THREADS = 4  # Number of parallel threads
    
    # Create extractor and run
    extractor = CC12MExtractor(
        repo_name=REPO_NAME,
        output_dir=OUTPUT_DIR,
        num_threads=NUM_THREADS
    )
    
    extractor.download_and_extract_all(start_num=START_NUM, end_num=END_NUM)
    """Main execution function."""
    # Configuration
    TAR_DIR = "/path/to/cc12m/tar/files"  # Change this to your tar files directory
    OUTPUT_DIR = "/path/to/output"  # Change this to your desired output directory
    START_NUM = 0  # Start number (e.g., 0 for cc12m-train-0000.tar)
    END_NUM = 5    # End number (e.g., 5 for cc12m-train-0005.tar)
    NUM_THREADS = 4  # Number of parallel threads
    
    # Create extractor and run
    extractor = CC12MExtractor(
        tar_dir=TAR_DIR,
        output_dir=OUTPUT_DIR,
        num_threads=NUM_THREADS
    )
    
    extractor.extract_all(start_num=START_NUM, end_num=END_NUM)


if __name__ == "__main__":
    main()
