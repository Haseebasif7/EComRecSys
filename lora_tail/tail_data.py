import os
import pandas as pd
import shutil
from pathlib import Path
from config import CSV_PATH, IMAGE_DIR
# Paths
script_dir = Path(__file__).parent
csv_path = CSV_PATH
image_dir = IMAGE_DIR

# Create output folders in the same directory as this script
watches_folder = script_dir / "watches"
bracelets_folder = script_dir / "bracelets"

# Create folders if they don't exist
watches_folder.mkdir(exist_ok=True)
bracelets_folder.mkdir(exist_ok=True)

print("="*70)
print("Copying raw images for watches and bracelets")
print("="*70)

# Read the CSV file
df = pd.read_csv(csv_path)
print(f"\nTotal samples in CSV: {len(df)}")

# Filter for watches and bracelets
watches_df = df[df['product_name'] == 'watches'].copy()
bracelets_df = df[df['product_name'] == 'bracelet'].copy()

print(f"\nWatches samples: {len(watches_df)}")
print(f"Bracelets samples: {len(bracelets_df)}")

# Function to copy images
def copy_images(df, output_folder, product_name):
    """Copy raw images from dataset to output folder"""
    copied = 0
    failed = 0
    
    for idx, row in df.iterrows():
        filename = row['file']
        source_path = os.path.join(image_dir, filename)
        dest_path = os.path.join(output_folder, filename)
        
        try:
            # Copy the raw image file without any transformations
            shutil.copy2(source_path, dest_path)
            copied += 1
            if copied % 50 == 0:
                print(f"  Copied {copied} {product_name} images...")
        except Exception as e:
            print(f"  Error copying {filename}: {e}")
            failed += 1
    
    return copied, failed

# Copy watches images
print(f"\n{'='*70}")
print("Copying watches images...")
print(f"{'='*70}")
watches_copied, watches_failed = copy_images(watches_df, watches_folder, "watches")
print(f"\n✓ Watches: {watches_copied} images copied, {watches_failed} failed")

# Copy bracelets images
print(f"\n{'='*70}")
print("Copying bracelets images...")
print(f"{'='*70}")
bracelets_copied, bracelets_failed = copy_images(bracelets_df, bracelets_folder, "bracelets")
print(f"\n✓ Bracelets: {bracelets_copied} images copied, {bracelets_failed} failed")

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Watches folder: {watches_folder}")
print(f"  - Images copied: {watches_copied}")
print(f"  - Failed: {watches_failed}")
print(f"\nBracelets folder: {bracelets_folder}")
print(f"  - Images copied: {bracelets_copied}")
print(f"  - Failed: {bracelets_failed}")
print(f"\nTotal images copied: {watches_copied + bracelets_copied}")
print(f"\n✓ All images are raw (no transformations applied)")
print(f"✓ Folders created in: {script_dir}")

