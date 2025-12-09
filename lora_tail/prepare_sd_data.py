"""
Prepare image-text pairs for Stable Diffusion training.

This script:
1. Reads prompts from prompts_SD.txt
2. Assigns random prompts from the appropriate category to each image
3. Creates a metadata.jsonl file compatible with HuggingFace datasets
"""

import os
import sys
import json
import random
import shutil
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
script_dir = Path(__file__).parent
prompts_file = script_dir / "prompts_SD.txt"
watches_folder = script_dir / "data" / "watches"
bracelets_folder = script_dir / "data" / "bracelets"
output_folder = script_dir / "train_data"
metadata_file = output_folder / "metadata.jsonl"

# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}


def parse_prompts(prompts_file):
    """Parse prompts from prompts_SD.txt file."""
    watch_prompts = []
    bracelet_prompts = []
    
    current_category = None
    
    with open(prompts_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('Watch:'):
                current_category = 'watch'
                continue
            elif line.startswith('Bracelet:'):
                current_category = 'bracelet'
                continue
            
            if current_category == 'watch' and line:
                watch_prompts.append(line)
            elif current_category == 'bracelet' and line:
                bracelet_prompts.append(line)
    
    return watch_prompts, bracelet_prompts


def get_image_files(folder):
    """Get all image files from a folder."""
    image_files = []
    if folder.exists():
        for file in folder.iterdir():
            if file.suffix in IMAGE_EXTENSIONS:
                image_files.append(file)
    return sorted(image_files)


def create_metadata(watch_prompts, bracelet_prompts):
    """Create metadata.jsonl file with image-text pairs."""
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    watch_images = get_image_files(watches_folder)
    bracelet_images = get_image_files(bracelets_folder)
    
    print(f"Found {len(watch_images)} watch images")
    print(f"Found {len(bracelet_images)} bracelet images")
    print(f"Watch prompts: {len(watch_prompts)}")
    print(f"Bracelet prompts: {len(bracelet_prompts)}")
    
    # Create metadata entries
    metadata_entries = []
    
    # Process watch images
    for img_path in watch_images:
        # Select random prompt from watch prompts
        prompt = random.choice(watch_prompts)
        
        # Copy image to output folder (or use relative path)
        img_filename = img_path.name
        output_img_path = output_folder / img_filename
        
        # Copy image if it doesn't exist
        if not output_img_path.exists():
            shutil.copy2(img_path, output_img_path)
        
        # Create metadata entry
        entry = {
            "file_name": img_filename,
            "text": prompt
        }
        metadata_entries.append(entry)
    
    # Process bracelet images
    for img_path in bracelet_images:
        # Select random prompt from bracelet prompts
        prompt = random.choice(bracelet_prompts)
        
        # Copy image to output folder
        img_filename = img_path.name
        output_img_path = output_folder / img_filename
        
        # Copy image if it doesn't exist
        if not output_img_path.exists():
            shutil.copy2(img_path, output_img_path)
        
        # Create metadata entry
        entry = {
            "file_name": img_filename,
            "text": prompt
        }
        metadata_entries.append(entry)
    
    # Write metadata.jsonl file
    with open(metadata_file, 'w') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total image-text pairs: {len(metadata_entries)}")
    print(f"  - Watches: {len(watch_images)}")
    print(f"  - Bracelets: {len(bracelet_images)}")
    print(f"\nMetadata file created: {metadata_file}")
    print(f"Training data folder: {output_folder}")
    print(f"\n✓ Data is ready for Stable Diffusion training!")
    print(f"✓ Use --train_data_dir {output_folder} in your training script")


if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    random.seed(42)
    
    # Parse prompts
    print("="*70)
    print("Preparing Stable Diffusion Training Data")
    print("="*70)
    print(f"\nReading prompts from: {prompts_file}")
    
    watch_prompts, bracelet_prompts = parse_prompts(prompts_file)
    
    if not watch_prompts:
        print("ERROR: No watch prompts found!")
        sys.exit(1)
    
    if not bracelet_prompts:
        print("ERROR: No bracelet prompts found!")
        sys.exit(1)
    
    # Create metadata
    create_metadata(watch_prompts, bracelet_prompts)

