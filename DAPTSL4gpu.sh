#!/bin/bash
#SBATCH --job-name=DAPTSLexample
#SBATCH --output=%x_%j.out    
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=05:00:00
#SBATCH --partition=gpu_a100
#SBATCH --constraint=scratch-node


# Load CUDA module (adjust version to match your system)
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source /home/svhonk/daptslenv/bin/activate  # activate your virtual environment 
export PYTHONPATH=.
torchrun --nproc_per_node=4 --master_port=29500 \
  dinov2/train/train.py \
  --config-file dinov2/configs/train/fmow_vitb144gpu.yaml \
  --output-dir "run4g/output" \
  train.num_workers=8
