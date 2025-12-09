import json, pandas as pd
from pathlib import Path
from config import CSV_PATH, IMAGE_DIR

df = pd.read_csv(CSV_PATH)              # e.g. filtered CSV
out_dir = Path("lora_tail/data/watches")  # change per split
records = []
for _, row in df.iterrows():
    file_name = row["file"]
    if (out_dir / file_name).exists():
        caption = f"a photo of a {row['product_name']} for ecommerce"
        records.append({"file_name": file_name, "text": caption})

with open(out_dir / "metadata.jsonl", "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")
print(f"wrote {len(records)} captions to {out_dir/'metadata.jsonl'}")
