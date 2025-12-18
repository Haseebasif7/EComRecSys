## EComRecSys – Tail Aware Visual Recommendation Toolkit

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-green.svg)](https://streamlit.io/)

**EComRecSys** is an end‑to‑end experimentation repo for e‑commerce visual recommendation on the **Style Color Images** dataset  
([Kaggle link](https://www.kaggle.com/datasets/olgabelitskaya/style-color-images)).  
It focuses on:
- Understanding **class imbalance** and **tail classes**
- **Augmenting tail classes** via Stable Diffusion + LoRA
- Building an **image similarity search** front‑end for product discovery

---

## High‑Level Flow

```text
                ┌────────────────────────────────────────┐
                │            Raw Dataset (CSV, Images)   │
                │          (CSV_PATH, IMAGE_DIR)         │
                └────────────────────────────────────────┘
                                   │
                                   ▼
                   ┌─────────────────────────────────┐
                   │ 1) Data Filtering & Class Stats │
                   │    - data/filter_csv.py         │
                   │    - data/class_distribution.py │
                   │    → identifies tail classes    │
                   └─────────────────────────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────────────────────────┐
              │ 2) Tail Class Extraction                        │
              │    - lora_tail/tail_data.py                     │
              │    → copies tail‑class images (e.g. watches,    │
              │      bracelets) into lora_tail/data/{class}/    │
              └─────────────────────────────────────────────────┘
                                   │
                                   ▼
      ┌─────────────────────────────────────────────────────────────────┐
      │ 3) Tail Data Augmentation with Stable Diffusion + LoRA         │
      │    - lora_tail/prepare_sd_data.py (builds image/text pairs)    │
      │    - lora_tail/train.sh (wrapper around HF LoRA script)        │
      │    → fine‑tunes SD on tail classes to synthetically augment    │
      │      rare categories                                          │
      └─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
           ┌────────────────────────────────────────────────┐
           │ 4) Image Embeddings & Vector Index             │
           │    - image_search/embedding_extractor.py       │
           │        (ResNet‑50 → 2048‑dim embeddings)       │
           │    - image_search/vector_db.py (FAISS index)   │
           │    - image_search/build_database.py            │
           │    → creates searchable vector DB over images  │
           └────────────────────────────────────────────────┘
                                   │
                                   ▼
          ┌─────────────────────────────────────────────────────┐
          │ 5) Visual Search Frontend (Streamlit)              │
          │    - image_search/app.py                           │
          │    - image_search/similarity_search.py             │
          │    → upload an image → ResNet‑50 → FAISS nearest   │
          │      neighbours → similar products UI              │
          └─────────────────────────────────────────────────────┘
```

---

## Repository Structure (key files)

```text
rs/
├── config.py                     # Paths: CSV_PATH, IMAGE_DIR
├── requirements.txt              # All Python dependencies
│
├── data/
│   ├── filter_csv.py             # Keeps only [file, product_name] columns
│   ├── style_dataset.py          # PyTorch dataset & transforms
│   ├── example_usage.py          # Example DataLoader & class stats
│   ├── class_distribution.py     # Class distribution plot + tail classes
│   └── seed.py                   # Reproducible seeding helper
│
├── lora_tail/
│   ├── TRAINING_GUIDE.md         # Detailed Stable Diffusion + LoRA training guide
│   ├── tail_data.py              # Copies tail‑class images to local folders
│   ├── prompts_SD.txt            # Training prompts for SD finetuning
│   ├── prepare_sd_data.py        # Builds image/text pairs + metadata.jsonl
│   ├── train_text_to_image_lora.py  # LoRA text‑to‑image training script (HF diffusers‑style)
│   └── train.sh                  # MacBook‑M‑friendly LoRA training wrapper
│
├── image_search/
│   ├── embedding_extractor.py    # ResNet‑50 encoder for image embeddings
│   ├── vector_db.py              # FAISS‑based vector database (cosine)
│   ├── similarity_search.py      # High‑level API for build/search
│   ├── build_database.py         # One‑shot embedding + index builder
│   ├── app.py                    # Streamlit UI for visual search
│   └── run_app.sh                # Helper launcher for the Streamlit app
│
└── lora_tail/train_data/         # (Generated) SD LoRA training data
```

---

## Setup & Installation

1. **Create and activate a virtual environment (optional but recommended)**  
   From the project root:
   ```bash
   python3.11 -m venv rs_env
   source rs_env/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure dataset paths**  
   Edit `config.py`:
   ```python
   CSV_PATH = "/absolute/path/to/style_filtered.csv"
   IMAGE_DIR = "/absolute/path/to/style_images_dir"
   ```

---

## Workflows

### 1. Inspect Class Distribution & Tail Classes

```bash
source rs_env/bin/activate
python data/filter_csv.py          # (only needed once, to create filtered CSV)
python data/class_distribution.py  # shows bar plot, 2 smallest classes in red
```

### 2. Prepare Tail Data for Stable Diffusion LoRA

```bash
source rs_env/bin/activate

python lora_tail/tail_data.py      # copies tail classes into lora_tail/data/
python lora_tail/prepare_sd_data.py
                                   # builds lora_tail/train_data/ + metadata
```

Then launch the Stable Diffusion LoRA training via the helper script  
(`lora_tail/train.sh`, which internally uses `lora_tail/train_text_to_image_lora.py`):

```bash
source rs_env/bin/activate

bash lora_tail/train.sh
```

> The LoRA training script and arguments can be customized inside `lora_tail/train.sh`.

### 3. Build Image Embedding Database

```bash
source rs_env/bin/activate

python image_search/build_database.py
```

This will:
- Load all images from `CSV_PATH` / `IMAGE_DIR`
- Extract ResNet‑50 embeddings
- Build and save a FAISS index + metadata under `vector_db/`

### 4. Run the Image Similarity Streamlit App

```bash
source rs_env/bin/activate

./image_search/run_app.sh
# or, equivalently:
# export KMP_DUPLICATE_LIB_OK=TRUE
# streamlit run image_search/app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`) and:
- Upload a query product image (e.g. watch / bracelet / shoe)
- See top‑K visually similar items with class labels and similarity scores

---

## Notes

- This repo focuses on **data prep, tail‑class identification, augmentation integration points,  
  and visual search**, and is designed to plug into an existing Stable Diffusion + LoRA training
  script from the Hugging Face **diffusers** ecosystem.
