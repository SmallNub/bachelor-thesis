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
# Ignore unused files

# 1. Make sure dest exists
mkdir -p "$TMPDIR"/bachelor-thesis

# 2. From your home dir, find all files/dirs except the ones to skip,
#    and copy them into $TMPDIR/bachelor-thesis, recreating their paths.
cd "$HOME"
find bachelor-thesis \
     \( -path 'bachelor-thesis/old' \
     -o -path 'bachelor-thesis/FinQA-main' \
     -o -path 'bachelor-thesis/GLEN-main' \) -prune \
     -o -print0 \
| xargs -0 -I{} cp --parents -R {} "$TMPDIR"/bachelor-thesis
