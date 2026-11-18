#!/bin/bash
#SBATCH --partition=bme_a10080g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=neurolm-learning
#SBATCH --output=/home_data/home/v-niuzhk/project/NeuroLM/data/runs/learning-%j.log
set -euo pipefail
ROOT=/home_data/home/v-niuzhk/project/NeuroLM
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TIKTOKEN_CACHE_DIR="$ROOT/data/tokenizer-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"
/home_data/home/v-niuzhk/anaconda3/envs/NeuroLM/bin/python -u scripts/learning_pipeline.py --steps 16
