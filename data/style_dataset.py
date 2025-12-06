import os
import sys
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CSV_PATH, IMAGE_DIR

class StyleImageDataset(Dataset):
    """
    PyTorch Dataset class for style color images.
    
    Args:
        csv_path (str): Path to the CSV file containing 'file' and 'product_name' columns
        image_dir (str): Directory containing the image files
        transform (callable, optional): Optional transform to be applied on images
    """
    
    def __init__(self, csv_path, image_dir, transform=None):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.transform = transform
        
        # Read the CSV file
        self.data = pd.read_csv(csv_path)
        
        # Verify required columns exist
        if 'file' not in self.data.columns or 'product_name' not in self.data.columns:
            raise ValueError("CSV must contain 'file' and 'product_name' columns")
        
        # Create label mapping for product_name
        self.product_names = self.data['product_name'].unique()
        self.product_to_idx = {name: idx for idx, name in enumerate(sorted(self.product_names))}
        self.idx_to_product = {idx: name for name, idx in self.product_to_idx.items()}
        
        # Add label column
        self.data['label'] = self.data['product_name'].map(self.product_to_idx)
        
        print(f"Dataset initialized with {len(self.data)} samples")
        print(f"Number of unique product names: {len(self.product_names)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get image filename and label
        img_filename = self.data.iloc[idx]['file']
        label = self.data.iloc[idx]['label']
        product_name = self.data.iloc[idx]['product_name']
        
        # Construct full image path
        img_path = os.path.join(self.image_dir, img_filename)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'product_name': product_name,
            'filename': img_filename
        }
    
    def get_num_classes(self):
        """Return the number of unique product name classes"""
        return len(self.product_names)
    
    def get_class_names(self):
        """Return list of all product name classes"""
        return sorted(self.product_names)


def get_default_transform(image_size=224, is_training=True):
    """
    Get default image transforms for training or validation.
    
    Args:
        image_size (int): Target image size (default: 224)
        is_training (bool): If True, apply data augmentation for training
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


if __name__ == "__main__":
    # Example usage
    csv_path = CSV_PATH
    image_dir = IMAGE_DIR
    
    # Create dataset with training transforms
    train_transform = get_default_transform(image_size=224, is_training=True)
    dataset = StyleImageDataset(csv_path, image_dir, transform=train_transform)
    
    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample loaded:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  Product name: {sample['product_name']}")
    print(f"  Filename: {sample['filename']}")
    
    # Create DataLoader example
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    print(f"\nDataLoader created with batch_size=32")
    print(f"Number of batches: {len(dataloader)}")

