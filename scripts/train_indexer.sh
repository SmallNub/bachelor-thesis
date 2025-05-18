#!/bin/bash
#SBATCH --job-name=train_finqa_indexer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --output=slurm/indexing_%j.out
#SBATCH --error=slurm/indexing_%j.err
#SBATCH --signal=USR1@300

source $HOME/bachelor-thesis/scripts/init_job.sh

# Change working directory
cd "$TMPDIR"/bachelor-thesis

function copy_to_home() {
    log INFO "Copying results to home directory"
    cp -rp "$TMPDIR"/bachelor-thesis/models $HOME/bachelor-thesis/
    cp -rp "$TMPDIR"/bachelor-thesis/logs $HOME/bachelor-thesis/
}

# Function to handle the signal
function _timeout_handler() {
    echo "Time limit approaching."
    copy_to_home
}

# Trap the USR1 signal
trap '_timeout_handler' USR1

# export PYTHONUNBUFFERED=1
deepspeed --num_gpus=$SLURM_GPUS_ON_NODE code/train_indexer.py --deepspeed ds_config.json

# Copy results back to home
copy_to_home
