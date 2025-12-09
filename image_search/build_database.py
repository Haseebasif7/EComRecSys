"""
Script to build the image similarity search database
Run this once to index all images
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from image_search.similarity_search import ImageSimilaritySearch


def main():
    print("="*70)
    print("Building Image Similarity Search Database")
    print("="*70)
    
    # Initialize search system
    search_system = ImageSimilaritySearch()
    
    # Build database
    search_system.build_database(
        save_path="vector_db",
        batch_size=32
    )
    
    print("\n✅ Database built successfully!")
    print("You can now use the Streamlit app to search for similar images.")
    print("Run: streamlit run image_search/app.py")


if __name__ == "__main__":
    main()

