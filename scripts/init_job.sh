#!/bin/bash

function log() {
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

    echo "[$timestamp] [$level] [$filename:$src_line:$src_func] $message"
}

log INFO "Script started"

# Load modules
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

log INFO "Modules loaded"

# Activate virtualenv
source $HOME/venv_glen/bin/activate

log INFO "Environment loaded"

# Copy data and code from home to compute
# Better I/O and preserve file structure
# Ignore unused files

mkdir -p "$TMPDIR"/bachelor-thesis

cp -r $HOME/bachelor-thesis/code "$TMPDIR"/bachelor-thesis
cp -r $HOME/bachelor-thesis/data "$TMPDIR"/bachelor-thesis
cp -r $HOME/bachelor-thesis/logs "$TMPDIR"/bachelor-thesis
cp -r $HOME/bachelor-thesis/models "$TMPDIR"/bachelor-thesis
cp -r $HOME/bachelor-thesis/scripts "$TMPDIR"/bachelor-thesis

log INFO "Initialization completed"
