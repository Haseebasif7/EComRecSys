"""
ResNet50 Embedding Extractor
Extracts feature embeddings from images using pre-trained ResNet50
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CSV_PATH, IMAGE_DIR


class ResNet50EmbeddingExtractor:
    """Extract embeddings from images using pre-trained ResNet50"""
    
    def __init__(self, device=None, embedding_dim=2048):
        """
        Initialize the ResNet50 embedding extractor
        
        Args:
            device: torch device (auto-detected if None)
            embedding_dim: Dimension of embeddings (2048 for ResNet50)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.embedding_dim = embedding_dim
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove the final classification layer to get embeddings
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"ResNet50 Embedding Extractor initialized on device: {self.device}")
    
    def extract_embedding(self, image_input):
        """
        Extract embedding from an image
        
        Args:
            image_input: Path to image file OR PIL Image object
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        try:
            # Handle both file path and PIL Image
            if isinstance(image_input, (str, Path)):
                # Load from file path
                try:
                    image = Image.open(image_input)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                except Exception as e:
                    print(f"Error opening image file {image_input}: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
            elif isinstance(image_input, Image.Image):
                # Already a PIL Image
                image = image_input
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            else:
                print(f"Unsupported image input type: {type(image_input)}")
                return None
            
            # Verify image is valid
            if image.size[0] == 0 or image.size[1] == 0:
                print(f"Error: Image has zero dimensions: {image.size}")
                return None
            
            print(f"Extracting embedding from image: size={image.size}, mode={image.mode}, device={self.device}")
            
            # Preprocess image
            try:
                image_tensor = self.transform(image).unsqueeze(0)
                print(f"Image tensor shape: {image_tensor.shape}")
                
                # Move to device
                image_tensor = image_tensor.to(self.device)
                print(f"Image tensor moved to device: {self.device}")
            except Exception as e:
                print(f"Error transforming image: {e}")
                import traceback
                traceback.print_exc()
                return None
            
            # Extract embedding
            try:
                print("Running model inference...")
                with torch.no_grad():
                    embedding = self.model(image_tensor)
                    print(f"Model output shape: {embedding.shape}")
                    
                    # Flatten the output (remove spatial dimensions)
                    embedding = embedding.view(embedding.size(0), -1)
                    print(f"Flattened embedding shape: {embedding.shape}")
                    
                    # Normalize the embedding (L2 normalization for cosine similarity)
                    embedding = nn.functional.normalize(embedding, p=2, dim=1)
                    print(f"Normalized embedding shape: {embedding.shape}")
                    
                    # Move to CPU and convert to numpy
                    embedding = embedding.cpu().numpy().flatten()
                    print(f"Final embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
                
                if embedding.shape[0] != self.embedding_dim:
                    print(f"Warning: Embedding dimension mismatch. Expected {self.embedding_dim}, got {embedding.shape[0]}")
                else:
                    print(f"✅ Successfully extracted embedding of dimension {embedding.shape[0]}")
                
                return embedding
            except Exception as e:
                print(f"Error extracting embedding: {e}")
                import traceback
                traceback.print_exc()
                return None
        except Exception as e:
            print(f"Unexpected error in extract_embedding: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_embeddings_batch(self, image_paths, batch_size=32):
        """
        Extract embeddings from multiple images in batches
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing
            
        Returns:
            numpy array of shape (num_images, embedding_dim)
        """
        embeddings = []
        valid_paths = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            batch_valid = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.transform(image)
                    batch_images.append(image_tensor)
                    batch_valid.append(path)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if batch_images:
                # Stack images into batch tensor
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                # Extract embeddings
                with torch.no_grad():
                    batch_embeddings = self.model(batch_tensor)
                    batch_embeddings = batch_embeddings.view(batch_embeddings.size(0), -1)
                    batch_embeddings = nn.functional.normalize(batch_embeddings, p=2, dim=1)
                    batch_embeddings = batch_embeddings.cpu().numpy()
                
                embeddings.append(batch_embeddings)
                valid_paths.extend(batch_valid)
        
        if embeddings:
            return np.vstack(embeddings), valid_paths
        else:
            return np.array([]), []
    
    def extract_from_dataset(self, csv_path, image_dir, batch_size=32):
        """
        Extract embeddings from all images in the dataset
        
        Args:
            csv_path: Path to CSV file with image filenames
            image_dir: Directory containing images
            batch_size: Batch size for processing
            
        Returns:
            tuple: (embeddings array, image paths, metadata dataframe)
        """
        import pandas as pd
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Build full image paths
        image_paths = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            img_path = os.path.join(image_dir, row['file'])
            if os.path.exists(img_path):
                image_paths.append(img_path)
                valid_indices.append(idx)
            else:
                print(f"Warning: Image not found: {img_path}")
        
        print(f"Found {len(image_paths)} valid images out of {len(df)} total")
        
        # Extract embeddings
        embeddings, valid_paths = self.extract_embeddings_batch(image_paths, batch_size)
        
        # Get metadata for valid images
        valid_df = df.iloc[valid_indices].copy()
        valid_df['full_path'] = valid_paths
        
        return embeddings, valid_paths, valid_df


if __name__ == "__main__":
    # Example usage
    extractor = ResNet50EmbeddingExtractor()
    
    # Extract embeddings from dataset
    embeddings, paths, metadata = extractor.extract_from_dataset(
        CSV_PATH, 
        IMAGE_DIR, 
        batch_size=32
    )
    
    print(f"\nExtracted {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Sample metadata:\n{metadata.head()}")

