#!/bin/bash
#SBATCH --job-name=train_finqa_indexer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_h100
#SBATCH --time=00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=steven.dong@student.uva.nl
#SBATCH --output=slurm/indexing_%j.out
#SBATCH --error=slurm/indexing_%j.err

source $HOME/bachelor-thesis/scripts/init_job.sh

# Change working directory
cd "$TMPDIR"/bachelor-thesis

deepspeed --num_gpus=1 code/train_indexer.py --deepspeed ds_config.json

# Copy results back to home
cp -rp "$TMPDIR"/bachelor-thesis/models $HOME/bachelor-thesis/
cp -rp "$TMPDIR"/bachelor-thesis/logs $HOME/bachelor-thesis/
