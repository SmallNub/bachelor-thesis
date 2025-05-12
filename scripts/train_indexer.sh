#!/bin/bash
#SBATCH --job-name=train_finqa_indexer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpu_h100
#SBATCH --time=00:30:00
#SBATCH --output=slurm/indexing_%j.out
#SBATCH --error=slurm/indexing_%j.err

source $HOME/bachelor-thesis/scripts/init_job.sh

# Change working directory
cd "$TMPDIR"/bachelor-thesis

export PYTHONUNBUFFERED=1
deepspeed --num_gpus=$SLURM_GPUS_ON_NODE code/train_indexer.py --deepspeed ds_config.json

# Copy results back to home
cp -rp "$TMPDIR"/bachelor-thesis/models $HOME/bachelor-thesis/
cp -rp "$TMPDIR"/bachelor-thesis/logs $HOME/bachelor-thesis/
