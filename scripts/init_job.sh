#!/bin/bash

# Load modules
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Copy virtualenv from home to compute and activate
# Prevents reading packages on home during runtime
cp -r $HOME/venv_glen "$TMPDIR"/venv_glen
source "$TMPDIR"/venv_glen/bin/activate

# Copy data and code from home to compute
# Better I/O and preserve file structure
cp -r $HOME/bachelor-thesis "$TMPDIR"
