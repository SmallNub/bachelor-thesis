#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source $HOME/venv_test/bin/activate

cp -r $HOME/bachelor-thesis "$TMPDIR"
cd "$TMPDIR"/bachelor-thesis/PAG-main

bash full_scripts/full_lexical_ripor_evaluate.sh

cp -r "$TMPDIR"/bachelor-thesis $HOME/test_output
