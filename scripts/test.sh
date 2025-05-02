#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00

source $HOME/bachelor-thesis/scripts/init_job.sh

# NQ320k

# Estimated duration: 9 minutes
bash scripts/eval_make_docid_glen_nq.sh

# Estimated duration: 12 minutes
bash scripts/eval_inference_query_glen_nq.sh

# MS MARCO

# Estimated duration: 20 hours
# bash scripts/eval_make_docid_glen_marco.sh

# Estimated duration: Unknown
# bash scripts/eval_inference_query_glen_marco.sh

# Copy results back to home
cp -r "$TMPDIR"/bachelor-thesis $HOME/test_output