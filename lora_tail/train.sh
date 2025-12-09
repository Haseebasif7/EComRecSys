#!/bin/bash

# Stable Diffusion LoRA Training Script
# Optimized for MacBook M4 Air (unified memory constraints)
# Dataset: 258 image-text pairs (watches + bracelets)
#
# Note: If you encounter MPS (Metal) errors with fp16, try:
#   - Remove --mixed_precision flag, or
#   - Change to --mixed_precision="bf16" (if supported)
#   - The training script automatically disables AMP for MPS if needed

# Configuration - Optimized for M4 Air
MODEL="runwayml/stable-diffusion-v1-5"
DATA_DIR="lora_tail/train_data"
OUTPUT_DIR="lora_tail/output"

# LoRA settings - Lower rank for memory efficiency
LORA_R=4                    # Lower rank = less memory (4-8 for M4)
LORA_ALPHA=16              # Typically 2-4x LoRA rank
LORA_DROPOUT=0.05

# Training settings - Conservative for unified memory
BATCH_SIZE=1                # Start with 1, increase if you have 16GB+ unified memory
GRADIENT_ACCUMULATION=16    # Simulates batch_size=16
EPOCHS=50                   # With 258 images, 50 epochs should be sufficient
LEARNING_RATE=1e-4          # Standard learning rate
RESOLUTION=512              # Standard SD resolution

# Memory optimization flags
# Note: gradient_checkpointing is a flag (no value needed)
MIXED_PRECISION="fp16"       # Use fp16 if MPS supports it, else remove this flag
DATALOADER_WORKERS=0         # macOS multiprocessing issues (0 recommended)

# Validation settings - Less frequent to save memory
VALIDATION_PROMPT="a detailed photograph of a watch"
VALIDATION_EPOCHS=10        # Validate every 10 epochs
NUM_VALIDATION_IMAGES=2     # Fewer images to save memory

# Other settings
CHECKPOINT_STEPS=500        # Save checkpoint every 500 steps
SEED=42

echo "=========================================="
echo "Starting LoRA Training for M4 Air"
echo "=========================================="
echo "Model: $MODEL"
echo "Data: $DATA_DIR (258 images)"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRADIENT_ACCUMULATION)))"
echo "Epochs: $EPOCHS"
echo "LoRA rank: $LORA_R, alpha: $LORA_ALPHA"
echo "=========================================="
echo ""

# Run training
python lora_tail/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="$MODEL" \
  --train_data_dir="$DATA_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --use_peft \
  --lora_r=$LORA_R \
  --lora_alpha=$LORA_ALPHA \
  --lora_dropout=$LORA_DROPOUT \
  --resolution=$RESOLUTION \
  --train_batch_size=$BATCH_SIZE \
  --num_train_epochs=$EPOCHS \
  --learning_rate=$LEARNING_RATE \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
  --gradient_checkpointing \
  --mixed_precision="$MIXED_PRECISION" \
  --dataloader_num_workers=$DATALOADER_WORKERS \
  --validation_prompt="$VALIDATION_PROMPT" \
  --num_validation_images=$NUM_VALIDATION_IMAGES \
  --validation_epochs=$VALIDATION_EPOCHS \
  --checkpointing_steps=$CHECKPOINT_STEPS \
  --seed=$SEED \
  --report_to="wandb"

echo ""
echo "=========================================="
echo "Training complete!"
echo "Check $OUTPUT_DIR for your LoRA weights"
echo "=========================================="

