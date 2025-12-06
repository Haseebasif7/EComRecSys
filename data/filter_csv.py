import pandas as pd
import os

# Path to the original CSV
csv_path = "/Users/haseeb/.cache/kagglehub/datasets/olgabelitskaya/style-color-images/versions/3/style/style.csv"

# Read the CSV
df = pd.read_csv(csv_path)

# Keep only 'file' and 'product_name' columns
df_filtered = df[['file', 'product_name']]

# Save the filtered CSV
output_path = "/Users/haseeb/.cache/kagglehub/datasets/olgabelitskaya/style-color-images/versions/3/style/style_filtered.csv"
df_filtered.to_csv(output_path, index=False)

print(f"Filtered CSV saved to: {output_path}")
print(f"Total rows: {len(df_filtered)}")
print(f"Columns: {df_filtered.columns.tolist()}")
print("\nFirst few rows:")
print(df_filtered.head())

