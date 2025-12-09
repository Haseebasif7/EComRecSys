# Stable Diffusion LoRA Training Guide

## Prerequisites

1. ✅ Data is prepared: `lora_tail/train_data/` with `metadata.jsonl`
2. ✅ All dependencies installed (see `requirements.txt`)
3. ✅ GPU with sufficient VRAM (recommended: 8GB+ for 512x512 images)

## Step-by-Step Training Instructions

### Step 1: Verify Your Data

Make sure your training data is ready:
```bash
# Check that metadata.jsonl exists and has entries
head -5 lora_tail/train_data/metadata.jsonl

# Count total images
ls lora_tail/train_data/*.png | wc -l
```

### Step 2: Choose a Base Model

Select a Stable Diffusion model from HuggingFace. Popular options:
- `runwayml/stable-diffusion-v1-5` (recommended for beginners)
- `stabilityai/stable-diffusion-2-1`
- `CompVis/stable-diffusion-v1-4`

### Step 3: Set Training Parameters

Key parameters to consider:

**Required:**
- `--pretrained_model_name_or_path`: Base SD model
- `--train_data_dir`: Path to your training data
- `--output_dir`: Where to save the trained LoRA

**LoRA-specific (use these flags):**
- `--use_peft`: Enable LoRA training
- `--lora_r`: LoRA rank (4-16, lower = smaller model, default: 4)
- `--lora_alpha`: LoRA alpha (typically 8-32, default: 32)
- `--lora_dropout`: Dropout rate (0.0-0.1, default: 0.0)

**Training parameters:**
- `--resolution`: Image resolution (512 or 768, default: 512)
- `--train_batch_size`: Batch size (1-4 depending on VRAM, default: 16)
- `--num_train_epochs`: Number of epochs (10-100, start with 50)
- `--learning_rate`: Learning rate (1e-4 to 1e-5, default: 1e-4)
- `--gradient_accumulation_steps`: Accumulate gradients (helps with small batch sizes)

**Memory optimization:**
- `--gradient_checkpointing`: Save memory (slower but uses less VRAM)
- `--mixed_precision`: Use fp16 or bf16 (saves memory)

**Optional but recommended:**
- `--validation_prompt`: Prompt for validation images
- `--validation_epochs`: How often to generate validation images
- `--report_to wandb`: Log to Weights & Biases

### Step 4: Run Training Command

**Basic training command (minimal):**
```bash
python lora_tail/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="lora_tail/train_data" \
  --output_dir="lora_tail/output" \
  --use_peft \
  --resolution=512 \
  --train_batch_size=1 \
  --num_train_epochs=50 \
  --learning_rate=1e-4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --validation_prompt="a detailed photograph of a watch" \
  --validation_epochs=5 \
  --report_to="wandb"
```

**Recommended command (balanced):**
```bash
python lora_tail/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="lora_tail/train_data" \
  --output_dir="lora_tail/output" \
  --use_peft \
  --lora_r=8 \
  --lora_alpha=32 \
  --lora_dropout=0.05 \
  --resolution=512 \
  --train_batch_size=2 \
  --num_train_epochs=100 \
  --learning_rate=1e-4 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --validation_prompt="a detailed photograph of a watch" \
  --num_validation_images=4 \
  --validation_epochs=10 \
  --checkpointing_steps=500 \
  --seed=42 \
  --report_to="wandb"
```

**Memory-constrained (for GPUs with <8GB VRAM):**
```bash
python lora_tail/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="lora_tail/train_data" \
  --output_dir="lora_tail/output" \
  --use_peft \
  --lora_r=4 \
  --lora_alpha=16 \
  --resolution=512 \
  --train_batch_size=1 \
  --num_train_epochs=50 \
  --learning_rate=1e-4 \
  --gradient_accumulation_steps=8 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --dataloader_num_workers=0
```

### Step 5: Monitor Training

- **Console output**: Watch for loss values decreasing
- **Weights & Biases**: If using `--report_to wandb`, check your dashboard
- **Validation images**: Check `output_dir` for generated validation images

### Step 6: Check Output

After training, your LoRA weights will be in:
```
lora_tail/output/
├── pytorch_lora_weights.safetensors  # LoRA weights
└── ... (other files)
```

### Step 7: Use Your Trained LoRA

Load and use your LoRA with:
```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.load_lora_weights("lora_tail/output")
pipe.to("cuda")

image = pipe("a detailed photograph of a watch").images[0]
image.save("generated_watch.png")
```

## Troubleshooting

**Out of Memory (OOM) errors:**
- Reduce `--train_batch_size` to 1
- Increase `--gradient_accumulation_steps`
- Enable `--gradient_checkpointing`
- Use `--mixed_precision="fp16"` or `"bf16"`
- Reduce `--resolution` to 512

**Training too slow:**
- Increase `--train_batch_size` if you have VRAM
- Reduce `--gradient_accumulation_steps`
- Disable `--gradient_checkpointing` (if you have enough VRAM)

**Poor results:**
- Increase `--num_train_epochs`
- Adjust `--learning_rate` (try 5e-5 or 2e-4)
- Increase `--lora_r` (try 8 or 16)
- Check your prompts in `metadata.jsonl`

## Quick Start (Copy-Paste Ready)

```bash
cd /Users/haseeb/Desktop/rs

python lora_tail/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="lora_tail/train_data" \
  --output_dir="lora_tail/output" \
  --use_peft \
  --lora_r=8 \
  --lora_alpha=32 \
  --resolution=512 \
  --train_batch_size=2 \
  --num_train_epochs=100 \
  --learning_rate=1e-4 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --validation_prompt="a detailed photograph of a watch" \
  --validation_epochs=10 \
  --seed=42 \
  --report_to="wandb"
```

