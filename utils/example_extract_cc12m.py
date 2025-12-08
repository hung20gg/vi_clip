"""
Example script to extract CC12M dataset from HuggingFace.
Modify the configuration section to match your setup.
"""

from extract_cc12m import CC12MExtractor
import time

# ===== CONFIGURATION =====
# HuggingFace repository name (e.g., "username/cc12m-dataset")
REPO_NAME = "pixparse/cc12m-wds"

# Directory where you want to save the extracted data
OUTPUT_DIR = "../../data/cc12m-raw"

# Range of tar files to extract (for testing, use 0-5)
START_NUM = 501  # cc12m-train-0000.tar
END_NUM = 1000    # cc12m-train-0001.tar
BATCH_SIZE = 25

# Number of parallel threads (adjust based on your CPU and network)
NUM_THREADS = 4

# Image resize dimension (images will be resized to RESIZE_TO x RESIZE_TO)
RESIZE_TO = 384
# =========================


def main():
    """Run the extraction."""
    
    print("=" * 60)
    print("CC12M Dataset Extractor (HuggingFace)")
    print("=" * 60)
    print(f"Repository: {REPO_NAME}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Extracting files: cc12m-train-{START_NUM:04d}.tar to cc12m-train-{END_NUM:04d}.tar")
    print(f"Threads: {NUM_THREADS}")
    print(f"Resize images to: {RESIZE_TO}x{RESIZE_TO}")
    print("=" * 60)
    print()
    
    for i in range(START_NUM, END_NUM, BATCH_SIZE):
        print(f"Preparing to extract: cc12m-train-{i:04d}.tar")
    # Create extractor
        extractor = CC12MExtractor(
            repo_name=REPO_NAME,
            output_dir=OUTPUT_DIR,
            num_threads=NUM_THREADS,
            resize_to=RESIZE_TO
        )
        
        start_time = time.time()
        # Run download and extraction
        extractor.download_and_extract_all(start_num=i, end_num=min(i + BATCH_SIZE - 1, END_NUM))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        print()
        print("=" * 60)
        print("Extraction Complete!")
        print(f"Images saved to: {OUTPUT_DIR}/images/")
        print(f"Captions saved to: {OUTPUT_DIR}/captions_{START_NUM:04d}_{END_NUM:04d}.parquet")
        print("=" * 60)


if __name__ == "__main__":
    main()
