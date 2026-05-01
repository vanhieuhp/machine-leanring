import kagglehub

# apikey = 'KGAT_b03f9822b30f3d35a001e6d4ca31a43d
# Download latest version
path = kagglehub.competition_download('house-prices-advanced-regression-techniques')

print("Path to competition files:", path)