#!/bin/bash
#SBATCH --job-name="training indexer"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=steven.dong@student.uva.nl
#SBATCH --output=logs/indexing_%j.out
#SBATCH --error=logs/indexing_%j.err

source $HOME/bachelor-thesis/scripts/init_job.sh

deepspeed code/train_indexer.py --deepspeed ds_config.json

# Copy results back to home
cp -rf "$TMPDIR"/bachelor-thesis/models $HOME/bachelor-thesis
