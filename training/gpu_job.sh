#!/bin/bash
#SBATCH --account=def-mushrifs
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000M               # memory per node
#SBATCH --time=0-03:00
module load python/3.12 
module load scipy-stack
source ~/.venvs/venv/bin/activate
python train_SGDoptimizer.py