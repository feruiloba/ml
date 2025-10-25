#!/bin/bash

# run_experiments.sh

# This script runs multiple Python training commands sequentially.

# Exit immediately if a command fails

set -e

# Define experiments

declare -a experiments=(
"python train.py --init_from='gpt2-medium' --lora_rank=16 --lora_alpha=64 --lora_dropout=0.05 --out_dir='gpt-lora-16' --device=cpu"
"python train.py --init_from='gpt2-medium' --lora_rank=128 --lora_alpha=512 --lora_dropout=0.05 --out_dir='gpt-lora-128' --device=cpu"
"python train.py --init_from='gpt2-medium' --lora_rank=196 --lora_alpha=784 --lora_dropout=0.05 --out_dir='gpt-lora-196' --device=cpu"
)

# Loop over all experiments

for cmd in "${experiments[@]}"; do
echo "======================================================"
echo "Running: $cmd"
echo "======================================================"
eval $cmd
echo "✅ Finished: $cmd"
done

echo "🎉 All experiments completed successfully."
