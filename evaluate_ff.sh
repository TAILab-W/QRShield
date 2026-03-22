#!/usr/bin/env bash
set -e

# model and data
ROOT_DIR="/root/code/data"
BENCHMARK_BASE_MODEL_PATH="${ROOT_DIR}/pretrained_model/stable-diffusion-2-1-base"
ARTIST_DIR="${ROOT_DIR}/artist_data/Vincent_van_Gogh"
POISON_DATA_DIR="${ARTIST_DIR}/poison"


# output directory
POISONED_MODEL_DIR="${ROOT_DIR}/finetuned_model"
FF_GENERATED_POISON_IMG_DIR="${ROOT_DIR}/generated_images/Vincent_van_Gogh"

############################
#        start execution
############################

echo "===== Step 1: Full Finetune poisoning ====="

accelerate launch train_text_to_image.py \
    --pretrained_model_name_or_path "${BENCHMARK_BASE_MODEL_PATH}" \
    --train_data_dir "${POISON_DATA_DIR}" \
    --use_ema \
    --center_crop \
    --random_flip \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --lr_scheduler "constant" \
    --resolution 512 \
    --seed 123456 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --mixed_precision "fp16" \
    --max_train_steps 1600 \
    --checkpointing_steps 5000 \
    --learning_rate 5e-6 \
    --max_grad_norm 1 \
    --lr_warmup_steps 0 \
    --output_dir "${POISONED_MODEL_DIR}"

echo "===== Step 2: Generate Poisoned Images ====="

python3 generate.py \
    --test_metadata_path "${ARTIST_DIR}/test/metadata.jsonl" \
    --model_dir "${POISONED_MODEL_DIR}" \
    --output_dir "${FF_GENERATED_POISON_IMG_DIR}" \
    --model_type "full"

echo "===== All Done ✅ ====="