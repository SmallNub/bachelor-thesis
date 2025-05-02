#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cp -r $HOME/venv_glen "$TMPDIR"/venv_glen
source "$TMPDIR"/venv_glen/bin/activate

cp -r $HOME/bachelor-thesis "$TMPDIR"
cd "$TMPDIR"/bachelor-thesis/GLEN-main

bash scripts/eval_make_docid_glen_nq.sh
bash scripts/eval_inference_query_glen_nq.sh

bash scripts/eval_make_docid_glen_marco.sh
bash scripts/eval_inference_query_glen_marco.sh

cp -r "$TMPDIR"/bachelor-thesis $HOME/test_output
