#!/bin/bash

# Streamlit App Launcher for Image Similarity Search
# Fixes OpenMP library conflict on macOS

# Set environment variable to fix OpenMP duplicate library issue
export KMP_DUPLICATE_LIB_OK=TRUE

# Run Streamlit app
streamlit run image_search/app.py

