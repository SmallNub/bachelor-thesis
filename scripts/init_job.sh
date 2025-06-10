#!/bin/bash

start_time=$(date +%s)

# Set this at the top of your script or configure it as needed
LOG_FILE="$HOME/bachelor-thesis/logs/bash_$SLURM_JOBID.log"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

# Logging function
# Function: log [INFO|WARNING|ERROR] <message>
# Logs to stdout with format [Time] [Level] [File] Message
log() {
    local level="$1"
    shift
    local message="$*"

    local timestamp
    timestamp=$(date "+%Y-%m-%d %H:%M:%S,%3N")

    local src_file="${BASH_SOURCE[1]}"
    local src_line="${BASH_LINENO[0]}"
    local src_func="${FUNCNAME[1]:-main}"

    local filename
    filename=$(basename "$src_file")

    local log_line="[$timestamp] [$level] [$filename:$src_line:$src_func] $message"

    # Print to stdout
    echo "$log_line"

    # Append to log file
    echo "$log_line" >> "$LOG_FILE"
}

log INFO "Script started"

# Load modules
log INFO "Loading modules..."

module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0

log INFO "Modules loaded"

# Activate virtualenv
log INFO "Loading environment..."

source $HOME/venv_thesis/bin/activate

log INFO "Environment loaded"

# Copy data and code from home to compute
# Better I/O and preserve file structure
# Ignore unused files
log INFO "Copying files..."

mkdir -p "$TMPDIR"/bachelor-thesis

cp -r $HOME/bachelor-thesis/code "$TMPDIR"/bachelor-thesis
cp -r $HOME/bachelor-thesis/data "$TMPDIR"/bachelor-thesis
cp -r $HOME/bachelor-thesis/models "$TMPDIR"/bachelor-thesis
cp -r $HOME/bachelor-thesis/scripts "$TMPDIR"/bachelor-thesis

log INFO "Files copied"

# Change working directory
cd "$TMPDIR"/bachelor-thesis

# Load utilities
source $HOME/bachelor-thesis/scripts/signal_utils.sh

end_time=$(date +%s)
elapsed=$((end_time - start_time))

printf -v duration "%02d:%02d:%02d" \
    $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))

log INFO "Initialization completed in $duration"
