import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
import argparse
from torch.utils.data import Dataset, DataLoader
import timm


# Add parent directory to path to import CLIPImage
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.model import CLIP


model_args = {
    'text_model': 'vinai/phobert-base-v2',
    'vision_model': 'vit_base_patch16_clip_224.dfn2b', #vit_base_patch16_clip_224.dfn2b, vit_base_patch16_siglip_224
    'max_length': 64,
    'model_type': 'siglip', # 'text_siglip' or 'text_clip'
    'pretrain': True,
    'projection_dim':768,
    'force_text_projection': False
}

def load_model(model_name='google/siglip-base-patch16-224', device='cuda'):
    """Load SigLIP model and processor"""
    print(f"Loading model: {model_name}")

    model_args['vision_model'] = model_name
    
    clip = CLIP(
        **model_args
    )
    clip = clip.to(device)
    clip.eval()
    # Extract vision model using CLIPImage
    
    
    # Clean up
    torch.cuda.empty_cache()
    return clip


class ImageDataset(Dataset):
    """Dataset for loading images with their paths and output paths"""
    
    def __init__(self, image_paths, output_paths):
        self.image_paths = image_paths
        self.output_paths = output_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        output_path = self.output_paths[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))
            
            # Process image
            
            return {
                'inputs': torch.tensor(np.array(image)).permute(2,0,1).float(),
                'image_path': image_path,
                'output_path': output_path,
                'success': True
            }
        except Exception as e:
            # Return a placeholder for failed images
            return {
                'inputs': None,
                'image_path': image_path,
                'output_path': output_path,
                'success': False,
                'error': str(e)
            }


def collate_fn(batch):
    """Custom collate function to handle batching"""
    # Separate successful and failed items
    successful = [item for item in batch if item['success']]
    failed = [item for item in batch if not item['success']]
    
    if not successful:
        return {'batch': None, 'failed': failed}
    
    # Stack tensors for successful items
    pixel_values = torch.stack([item['inputs'] for item in successful])
    
    return {
        'batch': pixel_values,
        'image_paths': [item['image_path'] for item in successful],
        'output_paths': [item['output_path'] for item in successful],
        'failed': failed
    }


def embed_dataset(
    dataset_root,
    output_root,
    model_name='google/siglip-base-patch16-224',
    device='cuda',
    batch_size=32,
    num_workers=4,
    folder_start=None,
    folder_end=None
):
    """
    Process all images in dataset and save embeddings maintaining structure.
    
    Structure:
    dataset_root/
        images/
            folder_x/
                <id>.jpg
    
    Output:
    output_root/
        numpy/
            folder_x/
                <id>.npy
    
    Args:
        folder_start: Start folder range (e.g., 0, 10, 51). If None, process all folders.
        folder_end: End folder range (e.g., 10, 50, 60). If None, process all folders.
        batch_size: Number of images to process in parallel
        num_workers: Number of worker threads for data loading (to avoid I/O wait)
    """
    
    # Load model
    model = load_model(model_name, device)
    
    # Setup paths
    images_dir = os.path.join(dataset_root, 'images')
    numpy_dir = os.path.join(output_root, 'numpy')
    
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    
    for root, dirs, files in os.walk(images_dir):
        # Filter folders based on range if specified
        if folder_start is not None or folder_end is not None:
            # Get the folder name relative to images_dir
            rel_dir = os.path.relpath(root, images_dir)
            
            # Check if this is a numbered folder (e.g., 0000, 0051)
            folder_name = os.path.basename(root)
            if folder_name.isdigit():
                folder_num = int(folder_name)
                
                # Skip if outside range
                if folder_start is not None and folder_num < folder_start:
                    continue
                if folder_end is not None and folder_num > folder_end:
                    continue
        
        for file in files:
            if any(file.endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    # Prepare output paths and filter already processed images
    image_paths_to_process = []
    output_paths_to_process = []
    skipped = 0
    
    for image_path in image_files:
        rel_path = os.path.relpath(image_path, images_dir)
        output_path = os.path.join(numpy_dir, os.path.splitext(rel_path)[0] + '.npy')
        
        if os.path.exists(output_path):
            skipped += 1
            continue
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        image_paths_to_process.append(image_path)
        output_paths_to_process.append(output_path)
    
    print(f"Found {len(image_files)} images total")
    print(f"Skipped {skipped} already processed images")
    print(f"Processing {len(image_paths_to_process)} images")
    
    if len(image_paths_to_process) == 0:
        print("No images to process!")
        return
    
    # Create dataset and dataloader
    dataset = ImageDataset(image_paths_to_process, output_paths_to_process)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Process batches
    processed = 0
    failed_count = 0
    
    for batch_data in tqdm(dataloader, desc="Processing batches"):
        # Handle failed images
        if batch_data['failed']:
            for failed_item in batch_data['failed']:
                print(f"Failed to load {failed_item['image_path']}: {failed_item.get('error', 'Unknown error')}")
                failed_count += 1
        
        # Process successful batch
        if batch_data['batch'] is not None:
            
            with torch.no_grad():
                embeddings = model.encode_image(batch_data['batch'].to(device))
                embeddings = embeddings.cpu().numpy()
            
            # Save embeddings
            for i, output_path in enumerate(batch_data['output_paths']):
                np.save(output_path, embeddings[i:i+1])
                processed += 1
    
    print(f"\nProcessing complete!")
    print(f"Processed: {processed}")
    print(f"Failed: {failed_count}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Total: {len(image_files)}")


def main():
    parser = argparse.ArgumentParser(description='Embed images using CLIP/SigLIP model')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Root directory of dataset (contains images/ folder)')
    parser.add_argument('--output_root', type=str, default=None,
                        help='Output directory for embeddings (default: same as dataset_root)')
    parser.add_argument('--model', type=str, default='google/siglip-base-patch16-224',
                        help='Model name from HuggingFace')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading')
    parser.add_argument('--folder_start', type=int, default=None,
                        help='Start folder number (e.g., 0, 10, 51)')
    parser.add_argument('--folder_end', type=int, default=None,
                        help='End folder number (e.g., 10, 50, 60)')
    
    args = parser.parse_args()
    
    # Use dataset_root as output_root if not specified
    output_root = args.output_root if args.output_root else args.dataset_root
    
    embed_dataset(
        dataset_root=args.dataset_root,
        output_root=output_root,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        folder_start=args.folder_start,
        folder_end=args.folder_end
    )


if __name__ == '__main__':
    main()
