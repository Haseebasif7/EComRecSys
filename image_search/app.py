"""
Streamlit App for Image Similarity Search
Beautiful and modern UI for searching similar images
"""

import os
# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_search.similarity_search import ImageSimilaritySearch


# Page configuration
st.set_page_config(
    page_title="Image Similarity Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .similarity-score {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2ecc71;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_search_system(db_path="vector_db"):
    """Load the image similarity search system (cached)"""
    try:
        search_system = ImageSimilaritySearch(db_path=db_path)
        if search_system.db.index is None or search_system.db.index.ntotal == 0:
            return None
        return search_system
    except Exception as e:
        st.error(f"Error loading search system: {e}")
        return None


def display_image_grid(images_data, cols=3):
    """Display images in a grid layout"""
    cols_list = st.columns(cols)
    
    for idx, img_data in enumerate(images_data):
        col = cols_list[idx % cols]
        
        with col:
            # Display image
            if os.path.exists(img_data['image_path']):
                img = Image.open(img_data['image_path'])
                st.image(img, width='stretch', caption=f"Rank #{img_data['rank']}")
                
                # Display metadata
                similarity = img_data.get('similarity', 0)
                st.markdown(f"<div class='similarity-score'>Similarity: {similarity:.3f}</div>", 
                           unsafe_allow_html=True)
                
                if 'product_name' in img_data:
                    st.caption(f"Category: {img_data['product_name']}")
                if 'file' in img_data:
                    st.caption(f"File: {img_data['file']}")
            else:
                st.error(f"Image not found: {img_data['image_path']}")


def main():
    # Header
    st.markdown('<div class="main-header">🔍 Image Similarity Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Find similar images using AI-powered search</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Database status
        db_path = st.text_input("Database Path", value="vector_db")
        
        # Load search system
        search_system = load_search_system(db_path)
        
        if search_system is None:
            st.warning("⚠️ Database not found or empty. Please build the database first.")
            st.info("""
            To build the database, run:
            ```python
            from image_search.similarity_search import ImageSimilaritySearch
            search = ImageSimilaritySearch()
            search.build_database()
            ```
            """)
            return
        
        st.success(f"✅ Database loaded: {search_system.db.index.ntotal} images indexed")
        
        # Search parameters
        st.header("🔎 Search Parameters")
        num_results = st.slider("Number of Results", min_value=1, max_value=20, value=10)
        
        # Filter by category (if metadata available)
        if search_system.db.metadata is not None and 'product_name' in search_system.db.metadata.columns:
            categories = ['All'] + sorted(search_system.db.metadata['product_name'].unique().tolist())
            selected_category = st.selectbox("Filter by Category", categories)
        else:
            selected_category = 'All'
    
    # Main content area
    st.header("📤 Upload Query Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image to find similar images in the database"
    )
    
    # Display uploaded image and search
    if uploaded_file is not None:
        # Display query image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            try:
                # Reset file pointer
                uploaded_file.seek(0)
                query_image = Image.open(uploaded_file)
                query_image.load()  # Load image data
                st.image(query_image, caption="Query Image", width='stretch')
            except Exception as e:
                st.error(f"Error displaying image: {e}")
                query_image = None
        
        with col2:
            st.info("""
            **How it works:**
            1. Your image is processed through a pre-trained ResNet50 model
            2. Feature embeddings are extracted
            3. Similar images are found using cosine similarity
            4. Results are ranked by similarity score
            """)
        
        # Search button
        if st.button("🔍 Search Similar Images", type="primary"):
            with st.spinner("Searching for similar images..."):
                try:
                    # Reset file pointer before processing
                    uploaded_file.seek(0)
                    
                    # Perform search
                    results = search_system.search_similar_from_upload(uploaded_file, k=num_results)
                    
                    # Debug info
                    if results is None:
                        st.error("❌ Failed to extract embedding from uploaded image.")
                        
                        # Try to get more info about the error
                        try:
                            uploaded_file.seek(0)
                            # Try to open the image to see if it's valid
                            test_img = Image.open(uploaded_file)
                            test_img.load()
                            img_info = f"Image loaded successfully. Size: {test_img.size}, Mode: {test_img.mode}"
                            
                            with st.expander("🔍 Debug Information"):
                                st.success("✅ Image file is valid")
                                st.write(f"**Image Info:** {img_info}")
                                st.write("**The issue is likely in embedding extraction.**")
                                st.write("Check the terminal/console for detailed error messages.")
                        except Exception as img_error:
                            with st.expander("🔍 Debug Information"):
                                st.error(f"❌ Image file error: {img_error}")
                                st.write("**The uploaded file cannot be read as an image.**")
                                st.write("**Try:**")
                                st.write("- Re-upload the image")
                                st.write("- Convert to JPG or PNG format")
                                st.write("- Check if file is corrupted")
                        
                        with st.expander("🔍 Troubleshooting"):
                            st.write("**Possible issues:**")
                            st.write("1. Image format not supported (try JPG or PNG)")
                            st.write("2. Image file is corrupted")
                            st.write("3. Image dimensions are invalid")
                            st.write("4. Memory issue during processing")
                            st.write("5. Device compatibility issue (MPS/CUDA)")
                            st.write("")
                            st.write("**Check terminal/console for detailed error messages**")
                        return
                    
                    if not results.get('results'):
                        st.warning("⚠️ No results returned from search.")
                        with st.expander("🔍 Debug Information"):
                            st.write(f"Database size: {search_system.db.index.ntotal} images")
                            st.write(f"Query embedding extracted: {results is not None}")
                            st.write("Try:")
                            st.write("1. Upload a different image")
                            st.write("2. Check if the database was built correctly")
                            st.write("3. Verify the image format is supported")
                        return
                    
                    # Filter by category if selected
                    original_count = len(results['results'])
                    if selected_category != 'All':
                        results['results'] = [
                            r for r in results['results'] 
                            if r.get('product_name') == selected_category
                        ]
                        
                        if not results['results']:
                            st.warning(f"⚠️ No results found for category '{selected_category}'. Showing all results instead.")
                            # Re-search without category filter
                            results = search_system.search_similar_from_upload(uploaded_file, k=num_results)
                    
                    if results and results['results']:
                        st.success(f"✅ Found {len(results['results'])} similar images!")
                        
                        # Display results
                        st.header("📊 Search Results")
                        
                        # Results summary
                        similarities = [r['similarity'] for r in results['results']]
                        avg_similarity = sum(similarities) / len(similarities)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Results Found", len(results['results']))
                        with col2:
                            st.metric("Avg Similarity", f"{avg_similarity:.3f}")
                        with col3:
                            st.metric("Best Match", f"{max(similarities):.3f}")
                        
                        # Display images in grid
                        st.subheader("🖼️ Similar Images")
                        display_image_grid(results['results'], cols=3)
                        
                        # Detailed results table
                        with st.expander("📋 Detailed Results"):
                            results_df = pd.DataFrame([
                                {
                                    'Rank': r['rank'],
                                    'File': r.get('file', 'N/A'),
                                    'Category': r.get('product_name', 'N/A'),
                                    'Similarity': f"{r['similarity']:.4f}",
                                    'Path': r['image_path']
                                }
                                for r in results['results']
                            ])
                            st.dataframe(results_df, width='stretch')
                    else:
                        st.error("No similar images found. Try a different image.")
                        
                except Exception as e:
                    st.error(f"Error during search: {e}")
                    st.exception(e)
    
    else:
        # Show example or instructions
        st.info("""
        👆 **Upload an image above to get started!**
        
        The system will:
        - Extract features from your image using ResNet50
        - Search through the indexed database
        - Return the most similar images ranked by similarity
        """)
        
        # Show database statistics
        if search_system:
            st.subheader("📈 Database Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Images", search_system.db.index.ntotal)
            
            if search_system.db.metadata is not None:
                with col2:
                    if 'product_name' in search_system.db.metadata.columns:
                        num_categories = search_system.db.metadata['product_name'].nunique()
                        st.metric("Categories", num_categories)
                
                with col3:
                    if 'product_name' in search_system.db.metadata.columns:
                        category_counts = search_system.db.metadata['product_name'].value_counts()
                        st.metric("Most Common", category_counts.index[0] if len(category_counts) > 0 else "N/A")


if __name__ == "__main__":
    main()

