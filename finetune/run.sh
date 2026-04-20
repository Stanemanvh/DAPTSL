#!/bin/bash                                                                                                
#SBATCH --partition=TODO                     # Request partition
#SBATCH --account=TODO
#SBATCH -J explora_finetune                  # Job name
#SBATCH -o outputs/explora_finetune%j.out    # output file (%j expands to jobID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 8                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=90000                          # server memory requested (per node)
#SBATCH -t 120:00:00                         # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:2                         # Type/number of GPUs needed
#SBATCH --constraint=16G                     # gpu memory


# If not using SLURM, set num_gpus manually
if [ -n "$SLURM_JOB_GPUS" ]; then
  num_gpus=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
else
  num_gpus=2
fi

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS"=$SLURM_JOB_GPUS

echo "working directory = "$SLURM_SUBMIT_DIR


base_dir="data_and_checkpoints" ;
base_csv_dir="${base_dir}/fmow_csvs" ;
base_experiment_dir="${base_dir}/explora_finetune" ;

dataset_type="fmow_hf" ;
train_csv="${base_csv_dir}/train_62classes.csv"
val_csv="${base_csv_dir}/test_62classes.csv"

lora_type="lora";
lora_rank=64 ;

model_type="vanilla" ;
input_size=224 ;
patch_size=14 ;
# patch_size=16 ;  # for MAE pretrained checkpoints
batch_size=64 ;
accum_iter=4 ;
epochs=12;
blr=1e-3 ;
hf_train_max_samples=30000 ;      # e.g., 50000; 0 disables absolute cap

# Make sure DINO checkpoints have "dino" in the name
pretrain_ckpt="${base_dir}/dinov2_vitb14_pretrain.pth"

out_dir="${base_experiment_dir}/finetune/run1/${lora_type}_dino_pretrain-blk11r64_${dataset_type}-finetune_${lora_rank}_bs128"

torchrun --nproc_per_node=1 --master_port=40001 finetune/finetune.py \
   --output_dir="${out_dir}" \
   --log_dir="${out_dir}" \
   --input_size="${input_size}" --patch_size="${patch_size}" \
   --batch_size="${batch_size}" --accum_iter="${accum_iter}" --num_workers 8 \
   --model vit_base_patch16 --epochs="${epochs}" --blr="${blr}" --layer_decay 1.0 \
   --weight_decay 0.0 --drop_path 0.2 --reprob 0 --mixup 0.0 --cutmix 0.0  \
   --lora_type="${lora_type}" --lora_rank="${lora_rank}" \
   --model_type="${model_type}" \
   --finetune="${pretrain_ckpt}" \
   --unfreeze_blocks 11 \
   --unfreeze_norm --unfreeze_cls_token \
   --save_every=30 \
   --dist_eval --num_workers 8 --dataset_type="${dataset_type}" \
   --train_path="${train_csv}" \
   --test_path="${val_csv}" \
#    --resume="${out_dir}/checkpoint-30.pth" \
#  --unfreeze_norm --unfreeze_cls_token --unfreeze_blocks 0 1 22 23
