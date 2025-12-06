import kagglehub

# Download latest version
path = kagglehub.dataset_download("olgabelitskaya/style-color-images")

print("Path to dataset files:", path)