import kagglehub

# Download latest version
## path to download
path = kagglehub.dataset_download("mehmoodsheikh/fairface-dataset")

print("Path to dataset files:", path)