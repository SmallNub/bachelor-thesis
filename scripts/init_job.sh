#!/bin/bash

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Script started"

# Load modules
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Modules loaded"

# Activate virtualenv
source $HOME/venv_glen/bin/activate

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Environment loaded"

# Copy data and code from home to compute
# Better I/O and preserve file structure
# Ignore unused files

mkdir -p "$TMPDIR"/bachelor-thesis

cp -r $HOME/bachelor-thesis/code "$TMPDIR"/bachelor-thesis
cp -r $HOME/bachelor-thesis/data "$TMPDIR"/bachelor-thesis
cp -r $HOME/bachelor-thesis/logs "$TMPDIR"/bachelor-thesis
cp -r $HOME/bachelor-thesis/models "$TMPDIR"/bachelor-thesis
cp -r $HOME/bachelor-thesis/scripts "$TMPDIR"/bachelor-thesis

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Initialization completed"
