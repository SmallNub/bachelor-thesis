#!/bin/bash
#SBATCH --job-name=eval_finqa_full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_h100
#SBATCH --time=6:00:00
#SBATCH --output=slurm/eval_%j.out
#SBATCH --error=slurm/eval_%j.err
#SBATCH --signal=USR1@300

source $HOME/bachelor-thesis/scripts/init_job.sh

# See Snellius known issues/FAQ
export NCCL_SOCKET_IFNAME="eno2np0"
export PYTHONUNBUFFERED=1

# Start the main process in the background for immediate signal handling
log INFO "Starting main process in the background"
start_time=$(date +%s)

python code/eval.py &
pid=$!

log INFO "Main process started with pid $pid"

# Wait for main process to finish while handling signals
wait_with_signals $pid

end_time=$(date +%s)
elapsed=$((end_time - start_time))

printf -v duration "%02d:%02d:%02d" \
    $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))

log INFO "Main process finished in $duration."

log INFO "Script finished"

# Wait shortly for buffers to flush
sleep 1
