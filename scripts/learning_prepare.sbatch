#!/bin/bash
#SBATCH --partition=bme_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=neurolm-prepare
#SBATCH --output=/home_data/home/v-niuzhk/project/NeuroLM/data/runs/prepare-%j.log
set -euo pipefail
ROOT=/home_data/home/v-niuzhk/project/NeuroLM
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$ROOT"
/home_data/home/v-niuzhk/anaconda3/envs/NeuroLM/bin/python -u dataset_maker/prepare_learning_data.py
