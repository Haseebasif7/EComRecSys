"""
Image Similarity Search System
Main interface for building and querying the image search system
"""

import os
import sys
import io
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from image_search.embedding_extractor import ResNet50EmbeddingExtractor
from image_search.vector_db import VectorDatabase
from config import CSV_PATH, IMAGE_DIR


class ImageSimilaritySearch:
    """Complete image similarity search system"""
    
    def __init__(self, db_path=None, device=None):
        """
        Initialize the image similarity search system
        
        Args:
            db_path: Path to saved vector database (if exists)
            device: torch device
        """
        self.extractor = ResNet50EmbeddingExtractor(device=device)
        self.db = VectorDatabase(embedding_dim=2048, index_type='cosine')
        
        if db_path and os.path.exists(db_path):
            print(f"Loading existing database from {db_path}")
            self.db.load(db_path)
        else:
            print("No existing database found. Build database first.")
    
    def build_database(self, csv_path=None, image_dir=None, save_path="vector_db", batch_size=32):
        """
        Build the vector database from images
        
        Args:
            csv_path: Path to CSV file (uses config if None)
            image_dir: Path to image directory (uses config if None)
            save_path: Where to save the database
            batch_size: Batch size for processing
        """
        csv_path = csv_path or CSV_PATH
        image_dir = image_dir or IMAGE_DIR
        
        print("="*70)
        print("Building Image Similarity Search Database")
        print("="*70)
        
        # Extract embeddings
        print("\nStep 1: Extracting embeddings from images...")
        embeddings, paths, metadata = self.extractor.extract_from_dataset(
            csv_path, 
            image_dir, 
            batch_size=batch_size
        )
        
        # Add to database
        print("\nStep 2: Adding embeddings to vector database...")
        self.db.add_embeddings(embeddings, paths, metadata)
        
        # Save database
        print(f"\nStep 3: Saving database to {save_path}...")
        self.db.save(save_path)
        
        print("\n" + "="*70)
        print("Database built successfully!")
        print(f"Total images indexed: {self.db.index.ntotal}")
        print("="*70)
    
    def search_similar(self, query_image_input, k=10):
        """
        Search for similar images
        
        Args:
            query_image_input: Path to query image OR PIL Image object
            k: Number of similar images to return
            
        Returns:
            dict with search results
        """
        # Extract embedding from query image
        query_embedding = self.extractor.extract_embedding(query_image_input)
        
        if query_embedding is None:
            return None
        
        # Search database
        distances, indices, paths, metadata = self.db.search(query_embedding, k=k)
        
        # Prepare results
        query_path = query_image_input if isinstance(query_image_input, (str, Path)) else "uploaded_image"
        results = {
            'query_path': query_path,
            'results': []
        }
        
        # For cosine similarity, the query image itself will typically have similarity ~1.0.
        # We usually don't want to return the exact query item as a result, so we skip the
        # top-1 result if its similarity is extremely close to 1.
        self_match_threshold = 0.9999

        for i, (dist, idx, path) in enumerate(zip(distances, indices, paths)):
            if i == 0 and dist >= self_match_threshold:
                continue

            result = {
                'rank': i + 1,
                'image_path': path,
                'similarity': float(dist),  # Cosine similarity (higher is better)
                'distance': float(dist)
            }
            
            # Add metadata if available
            if metadata is not None:
                row = metadata.iloc[i]
                result['file'] = row.get('file', os.path.basename(path))
                result['product_name'] = row.get('product_name', 'Unknown')
                if 'full_path' in row:
                    result['full_path'] = row['full_path']
            
            results['results'].append(result)
        
        return results
    
    def search_similar_from_upload(self, uploaded_image, k=10):
        """
        Search for similar images from uploaded image (for Streamlit)
        
        Args:
            uploaded_image: PIL Image or file-like object (Streamlit UploadedFile)
            k: Number of similar images to return
            
        Returns:
            dict with search results
        """
        # Handle different input types
        try:
            if isinstance(uploaded_image, Image.Image):
                # Already a PIL Image - use directly
                img = uploaded_image
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            else:
                # File-like object (Streamlit UploadedFile)
                try:
                    # Reset file pointer to beginning
                    uploaded_image.seek(0)
                    
                    # Read image data
                    img_bytes = uploaded_image.read()
                    
                    if len(img_bytes) == 0:
                        print("Error: Uploaded file is empty")
                        return None
                    
                    print(f"Read {len(img_bytes)} bytes from uploaded file")
                    
                    # Open from bytes
                    img = Image.open(io.BytesIO(img_bytes))
                    print(f"Image opened: size={img.size}, mode={img.mode}")
                    
                    # Load image to ensure it's fully decoded
                    img.load()
                    
                    # Convert to RGB if needed (required for ResNet50)
                    if img.mode != 'RGB':
                        print(f"Converting image from {img.mode} to RGB")
                        img = img.convert('RGB')
                    
                except Exception as e:
                    print(f"Error reading uploaded image: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
            
            # Verify image dimensions
            if img.size[0] == 0 or img.size[1] == 0:
                print(f"Error: Image has invalid dimensions: {img.size}")
                return None
            
            print(f"Processing image: size={img.size}, mode={img.mode}")
                
        except Exception as e:
            print(f"Error processing uploaded image: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Extract embedding directly from PIL Image (no temp file needed!)
        query_embedding = self.extractor.extract_embedding(img)
        
        if query_embedding is None:
            print("Error: Failed to extract embedding from uploaded image")
            return None
        
        if query_embedding.shape[0] != self.db.embedding_dim:
            print(f"Error: Embedding dimension mismatch. Expected {self.db.embedding_dim}, got {query_embedding.shape[0]}")
            return None
        
        # Search database
        try:
            distances, indices, paths, metadata = self.db.search(query_embedding, k=k)
        except Exception as e:
            print(f"Error during database search: {e}")
            return None
        
        # Check if we got any results
        if len(distances) == 0 or len(paths) == 0:
            print("Warning: Search returned empty results")
            return {'query_path': 'uploaded_image', 'results': []}
        
        # Prepare results
        results = {
            'query_path': 'uploaded_image',
            'results': []
        }

        # Same self-match skipping logic as in search_similar
        self_match_threshold = 0.9999

        for i, (dist, idx, path) in enumerate(zip(distances, indices, paths)):
            # Skip likely self-match (query image itself, if it also exists in the index)
            if i == 0 and dist >= self_match_threshold:
                continue

            # Skip invalid indices (FAISS can return -1 for empty results)
            if idx < 0 or idx >= len(self.db.image_paths):
                continue
                
            result = {
                'rank': len(results['results']) + 1,
                'image_path': path,
                'similarity': float(dist),  # Cosine similarity (higher is better)
                'distance': float(dist)
            }
            
            # Add metadata if available
            if metadata is not None and i < len(metadata):
                row = metadata.iloc[i]
                result['file'] = row.get('file', os.path.basename(path))
                result['product_name'] = row.get('product_name', 'Unknown')
                if 'full_path' in row:
                    result['full_path'] = row['full_path']
            else:
                result['file'] = os.path.basename(path)
                result['product_name'] = 'Unknown'
            
            results['results'].append(result)
        
        return results


if __name__ == "__main__":
    # Build database
    search_system = ImageSimilaritySearch()
    search_system.build_database(save_path="vector_db", batch_size=32)
    
    # Example search (if you have a test image)
    # results = search_system.search_similar("path/to/test/image.jpg", k=5)
    # print(results)

