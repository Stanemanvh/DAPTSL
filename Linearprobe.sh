#!/bin/bash
#SBATCH --job-name=finetune30ksamplesfmow
#SBATCH --output=%x_%j.out    
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_a100


# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS"=$SLURM_JOB_GPUS

echo "working directory = "$SLURM_SUBMIT_DIR

# ============ Paths ============
base_dir="data_and_checkpoints"
base_csv_dir="${base_dir}/fmow_csvs"
base_experiment_dir="${base_dir}/explora_linprobe"

# ============ Config ============
# cfg_file="linprobe/configs/fmow_vitb14.yaml"
cfg_file="linprobe/configs/fmow_vitl14.yaml"
# cfg_file="linprobe/configs/fmow_vitg14.yaml"

# ============ Pretrained weights ============
# Vanilla DINOv2 pretrained weights. If using this, set the cfg_file to be linprobe/configs/vitl14_pretrain.yaml
# pretrained_weights="${base_dir}/dinov2_vitl14_pretrain.pth"

# ExPLoRA pretrained weights
pretrained_weights="${base_dir}/explora_dinov2_vit_large_fmow_rgb_encoder_only.pth"

# ============ Output directory ============
out_dir="${base_experiment_dir}/lp30konpretrained"
hf_train_max_samples=0  # e.g., 30000; 0 disables cap

# ============ Run ============
export PYTHONPATH=.
WANDB__SERVICE_WAIT=300 torchrun --nproc_per_node=1 --master_port=40001 -m linprobe.linear \
    --wandb=explora_linprobe \
    --wandb_entity=TODO \
    --train-dataset="fmow:data_and_checkpoints/fmow_csvs/train_62classes.csv" \
    --val-dataset="fmow:data_and_checkpoints/fmow_csvs/test_62classes.csv" \
    --config-file="${cfg_file}" \
    --output-dir="${out_dir}" \
    --pretrained-weights="${pretrained_weights}" \
    --batch-size=256 \
    --hf-train-max-samples=${hf_train_max_samples} \
    --epochs=10 \
    --epoch-length=1500 \
    --save-checkpoint-frequency=1500 \
    --eval-period-iterations=1500
