#!/bin/bash
#SBATCH --job-name=train_finqa_full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_h100
#SBATCH --time=48:00:00
#SBATCH --output=slurm/training_%j.out
#SBATCH --error=slurm/training_%j.err
#SBATCH --signal=USR1@300

source $HOME/bachelor-thesis/scripts/init_job.sh

model_name="finqa_full_base"

copy_to_home() {
    log INFO "Copying results to home directory"
    local src_model="$TMPDIR/bachelor-thesis/models/$model_name"
    local dst_model="$HOME/bachelor-thesis/models/"

    local max_retries=3
    local timeout_secs=300  # 5 minutes

    try_rsync() {
        local src=$1
        local dst=$2
        local name=$3

        for attempt in $(seq 1 $max_retries); do
            log INFO "[$name] Attempt $attempt of $max_retries"
            if timeout $timeout_secs rsync -a "$src" "$dst"; then
                log INFO "[$name] Copy successful"
                return 0
            else
                log WARNING "[$name] Copy attempt $attempt failed"
                sleep 2
            fi
        done

        log ERROR "[$name] Failed to copy after $max_retries attempts"
        return 1
    }

    try_rsync "$src_model" "$dst_model" "Model"

    log INFO "Results copied successfully"
}

__on_signal() {
    local signal=$1
    log INFO "Processing signal $signal"
    copy_to_home
}

# See Snellius known issues/FAQ
export NCCL_SOCKET_IFNAME="eno2np0"
export PYTHONUNBUFFERED=1

# Start the main process in the background for immediate signal handling
log INFO "Starting main process in the background"
start_time=$(date +%s)

# deepspeed --num_gpus=$SLURM_GPUS_ON_NODE code/train_full.py &
python code/train_full.py &
pid=$!

log INFO "Main process started with pid $pid"

# Wait for main process to finish while handling signals
wait_with_signals $pid

end_time=$(date +%s)
elapsed=$((end_time - start_time))

printf -v duration "%02d:%02d:%02d" \
    $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))

log INFO "Main process finished in $duration."

# Copy results back to home
copy_to_home

log INFO "Script finished"

# Wait shortly for buffers to flush
sleep 1
