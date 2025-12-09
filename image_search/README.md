# Image Similarity Search System

A complete image similarity search pipeline using ResNet50 embeddings and FAISS vector database, with a beautiful Streamlit frontend.

## Features

- 🎯 **Pre-trained ResNet50**: Uses ImageNet pre-trained ResNet50 for feature extraction
- 🔍 **FAISS Vector Database**: Fast and efficient similarity search using FAISS
- 🎨 **Streamlit UI**: Beautiful and modern web interface
- 📊 **Metadata Support**: Stores and filters by product categories
- ⚡ **Batch Processing**: Efficient batch processing for large datasets
- 🚀 **Optimized**: L2 normalization for cosine similarity search

## Architecture

```
Image → ResNet50 → Embeddings (2048-dim) → FAISS Index → Similarity Search
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: For GPU support with FAISS, use `faiss-gpu` instead of `faiss-cpu`.

## Quick Start

### Step 1: Build the Database

First, extract embeddings from all images and build the vector database:

```bash
python image_search/build_database.py
```

This will:
- Load images from your CSV file (defined in `config.py`)
- Extract ResNet50 embeddings from all images
- Build and save the FAISS index
- Store metadata for filtering

### Step 2: Launch the Streamlit App

```bash
streamlit run image_search/app.py
```

The app will open in your browser where you can:
- Upload query images
- Search for similar images
- Filter by category
- View similarity scores

## Usage

### Python API

```python
from image_search.similarity_search import ImageSimilaritySearch

# Load existing database
search_system = ImageSimilaritySearch(db_path="vector_db")

# Search for similar images
results = search_system.search_similar("path/to/query/image.jpg", k=10)

# Process results
for result in results['results']:
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Image: {result['image_path']}")
    print(f"Category: {result.get('product_name', 'N/A')}")
```

### Building Database Programmatically

```python
from image_search.similarity_search import ImageSimilaritySearch

# Initialize
search_system = ImageSimilaritySearch()

# Build database
search_system.build_database(
    csv_path="path/to/data.csv",
    image_dir="path/to/images",
    save_path="vector_db",
    batch_size=32
)
```

## File Structure

```
image_search/
├── embedding_extractor.py    # ResNet50 embedding extraction
├── vector_db.py              # FAISS vector database
├── similarity_search.py      # Main search interface
├── build_database.py         # Database building script
├── app.py                    # Streamlit web app
└── README.md                 # This file
```

## Configuration

Update `config.py` with your data paths:

```python
CSV_PATH = "path/to/your/filtered.csv"
IMAGE_DIR = "path/to/your/images"
```

The CSV should have columns: `file` and `product_name` (or `category`).

## Performance

- **Embedding Extraction**: ~100-200 images/second (depends on hardware)
- **Search Speed**: <10ms for 10,000 images
- **Memory**: ~8MB per 1000 images (2048-dim embeddings)

## Advanced Features

### Custom Embedding Dimensions

The system uses ResNet50 which produces 2048-dimensional embeddings. You can modify this in `embedding_extractor.py` if using a different model.

### Index Types

- **Cosine Similarity** (default): Best for normalized embeddings
- **L2 Distance**: Alternative distance metric

Change in `vector_db.py`:
```python
db = VectorDatabase(embedding_dim=2048, index_type='cosine')  # or 'L2'
```

### Batch Size Optimization

Adjust batch size based on your GPU memory:
- GPU: 32-64
- CPU/MPS: 8-16

## Troubleshooting

### Database Not Found
Run `build_database.py` first to create the index.

### Out of Memory
- Reduce batch size in `build_database.py`
- Use `faiss-cpu` instead of `faiss-gpu`

### Slow Search
- Ensure you're using cosine similarity with normalized embeddings
- Consider using GPU-accelerated FAISS (`faiss-gpu`)

## Future Improvements

- [ ] Support for multiple embedding models
- [ ] GPU-accelerated FAISS
- [ ] Real-time database updates
- [ ] Advanced filtering options
- [ ] Search history and analytics
- [ ] Export search results

## License

Same as the main project.

