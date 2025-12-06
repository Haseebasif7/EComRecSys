import pandas as pd
import os
from config import CSV_PATH, IMAGE_DIR
# Path to the original CSV
csv_path = CSV_PATH

# Read the CSV
df = pd.read_csv(csv_path)

# Keep only 'file' and 'product_name' columns
df_filtered = df[['file', 'product_name']]

# Save the filtered CSV
output_path = CSV_PATH.replace(".csv", "_filtered.csv")
df_filtered.to_csv(output_path, index=False)

print(f"Filtered CSV saved to: {output_path}")
print(f"Total rows: {len(df_filtered)}")
print(f"Columns: {df_filtered.columns.tolist()}")
print("\nFirst few rows:")
print(df_filtered.head())

