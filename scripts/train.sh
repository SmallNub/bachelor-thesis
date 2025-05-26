#!/bin/bash
#SBATCH --job-name=train_finqa_full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_h100
#SBATCH --time=6:00:00
#SBATCH --output=slurm/training_%j.out
#SBATCH --error=slurm/training_%j.err
#SBATCH --signal=USR1@300

source $HOME/bachelor-thesis/scripts/init_job.sh

function copy_to_home() {
    log INFO "Copying results to home directory"
    cp -rp "$TMPDIR"/bachelor-thesis/models $HOME/bachelor-thesis/
    cp -rp "$TMPDIR"/bachelor-thesis/logs $HOME/bachelor-thesis/
    log INFO "Results copied successfully"
}

__on_signal() {
    local signal=$1
    log INFO "Processing signal $signal"
    copy_to_home
}

# See Snellius known issues/FAQ
export NCCL_SOCKET_IFNAME="eno2np0"
# export PYTHONUNBUFFERED=1

# Start the main process in the background for immediate signal handling
log INFO "Starting main process in the background"
start_time=$(date +%s)

deepspeed --num_gpus=$SLURM_GPUS_ON_NODE code/train_full.py &
pid=$!

log INFO "Main process started with pid $pid"

# Wait for main process to finish while handling signals
wait_with_signals $pid

end_time=$(date +%s)
elapsed=$((end_time - start_time))

printf -v duration "%02d:%02d:%02d" \
    $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))

log INFO "Main process finished in $duration. Interrupted? $__interrupted"

# Copy results back to home
copy_to_home

log INFO "Script finished"
