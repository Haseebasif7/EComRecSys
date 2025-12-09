"""
Vector Database for storing and searching image embeddings
Uses FAISS for efficient similarity search
"""

import os
import pickle
import numpy as np
import faiss
from pathlib import Path
import pandas as pd


class VectorDatabase:
    """FAISS-based vector database for image embeddings"""
    
    def __init__(self, embedding_dim=2048, index_type='L2'):
        """
        Initialize the vector database
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: 'L2' for L2 distance or 'cosine' for cosine similarity
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.metadata = None
        self.image_paths = None
        self.is_trained = False
        
    def create_index(self, num_vectors=None):
        """
        Create FAISS index
        
        Args:
            num_vectors: Expected number of vectors (for optimization)
        """
        if self.index_type == 'cosine':
            # For cosine similarity, use inner product on normalized vectors
            # FAISS uses inner product for normalized vectors
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            # L2 distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        self.is_trained = True
        print(f"Created {self.index_type} index with dimension {self.embedding_dim}")
    
    def add_embeddings(self, embeddings, image_paths, metadata=None):
        """
        Add embeddings to the index
        
        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            image_paths: List of image file paths
            metadata: Optional pandas DataFrame with metadata
        """
        if self.index is None:
            self.create_index()
        
        # Ensure embeddings are float32 for FAISS
        embeddings = embeddings.astype('float32')
        
        # Normalize for cosine similarity if needed
        if self.index_type == 'cosine':
            faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        self.image_paths = list(image_paths)
        if metadata is not None:
            self.metadata = metadata.copy()
        else:
            # Create basic metadata
            self.metadata = pd.DataFrame({
                'file': [os.path.basename(p) for p in image_paths],
                'path': image_paths
            })
        
        print(f"Added {len(embeddings)} embeddings to index")
        print(f"Total vectors in index: {self.index.ntotal}")
    
    def search(self, query_embedding, k=10):
        """
        Search for similar images
        
        Args:
            query_embedding: numpy array of shape (embedding_dim,)
            k: Number of similar images to return
            
        Returns:
            tuple: (distances, indices, paths, metadata)
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index is empty. Add embeddings first.")
        
        # Ensure query is float32 and reshape
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Normalize for cosine similarity if needed
        if self.index_type == 'cosine':
            faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Get results
        result_indices = indices[0]
        result_distances = distances[0]
        result_paths = [self.image_paths[i] for i in result_indices]
        
        # Get metadata
        if self.metadata is not None and len(result_indices) > 0:
            # Filter out invalid indices (-1 from FAISS)
            valid_mask = result_indices >= 0
            if valid_mask.any():
                valid_indices = result_indices[valid_mask]
                result_metadata = self.metadata.iloc[valid_indices].copy()
                result_metadata['distance'] = result_distances[valid_mask]
                # For cosine similarity, distance is already similarity (inner product of normalized vectors)
                # For L2, convert distance to similarity
                result_metadata['similarity'] = result_distances[valid_mask] if self.index_type == 'cosine' else (1 / (1 + result_distances[valid_mask]))
            else:
                result_metadata = None
        else:
            result_metadata = None
        
        return result_distances, result_indices, result_paths, result_metadata
    
    def save(self, save_path):
        """
        Save the index and metadata to disk
        
        Args:
            save_path: Directory path to save the database
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = save_path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        if self.metadata is not None:
            metadata_path = save_path / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'image_paths': self.image_paths,
                    'embedding_dim': self.embedding_dim,
                    'index_type': self.index_type
                }, f)
        
        print(f"Saved vector database to {save_path}")
    
    def load(self, load_path):
        """
        Load the index and metadata from disk
        
        Args:
            load_path: Directory path to load the database from
        """
        load_path = Path(load_path)
        
        # Load FAISS index
        index_path = load_path / "faiss_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = load_path / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.image_paths = data['image_paths']
                self.embedding_dim = data.get('embedding_dim', self.embedding_dim)
                self.index_type = data.get('index_type', self.index_type)
        
        self.is_trained = True
        print(f"Loaded vector database from {load_path}")
        print(f"Total vectors in index: {self.index.ntotal}")


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from image_search.embedding_extractor import ResNet50EmbeddingExtractor
    from config import CSV_PATH, IMAGE_DIR
    
    # Initialize extractor and database
    extractor = ResNet50EmbeddingExtractor()
    db = VectorDatabase(embedding_dim=2048, index_type='cosine')
    
    # Extract embeddings
    from config import CSV_PATH, IMAGE_DIR
    embeddings, paths, metadata = extractor.extract_from_dataset(
        CSV_PATH, 
        IMAGE_DIR, 
        batch_size=32
    )
    
    # Add to database
    db.add_embeddings(embeddings, paths, metadata)
    
    # Save database
    db.save("vector_db")

