"""
Example usage of the StyleImageDataset for training.

This script demonstrates how to:
1. Load the dataset
2. Create train/validation splits
3. Use DataLoaders for training
"""

from style_dataset import StyleImageDataset, get_default_transform
from torch.utils.data import DataLoader, random_split
import torch
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CSV_PATH, IMAGE_DIR

def show_tensor_image(tensor):
    from PIL import Image
    import numpy as np
    
    # CHW → HWC if torch tensor
    if hasattr(tensor, "permute"):
        tensor = tensor.permute(1, 2, 0).numpy()
    
    # Convert float to uint8
    if tensor.dtype != np.uint8:
        tensor = (tensor * 255).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.show()

def get_class_distribution(dataset):
    class_distribution = {}
    for i in dataset.get_class_names():
        class_distribution[i] = len(dataset.data[dataset.data['product_name'] == i])

    plt.figure(figsize=(10, 5))
    for i in class_distribution.keys():
        if class_distribution[i] < 140:
            plt.bar(i, class_distribution[i],width=0.5,align='center',color='red')
        else:
            plt.bar(i, class_distribution[i],width=0.5,align='center',color='green')
    plt.show()
    return class_distribution

def main():
    # Dataset paths
    split_ratio = 0.9
    csv_path = CSV_PATH
    image_dir = IMAGE_DIR

    # Create datasets with appropriate transforms
    train_transform = get_default_transform(image_size=224, is_training=True)
    val_transform = get_default_transform(image_size=224, is_training=False)

    # Create full dataset
    full_dataset = StyleImageDataset(csv_path, image_dir, transform=train_transform)

    # Split into train and validation by split_ratio
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Update validation dataset to use validation transforms
    val_dataset.dataset.transform = val_transform

    # Set num_workers based on platform (0 for macOS/Windows, 4 for Linux)
    # On macOS, multiprocessing with spawn can cause issues, so use 0
    num_workers = 0 if sys.platform == 'darwin' or sys.platform == 'win32' else 4

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Class distribution: {get_class_distribution(full_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {full_dataset.get_num_classes()}")
    print(f"Class names: {full_dataset.get_class_names()}")

    # Example: Iterate through a batch
    print("\nExample batch from training loader:")
    for batch in train_loader:
        images = batch['image']
        labels = batch['label']
        product_names = batch['product_name']
        
        print(f"Batch image shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"Sample product names: {product_names[:5]}")
        #show_tensor_image(images[28])
        break

    print("\nDataset is ready for training!")


if __name__ == '__main__':
    main()

