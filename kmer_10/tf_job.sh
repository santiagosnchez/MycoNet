#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=23:00:00

module load anaconda3
source activate tf_gpu2.1
python -B kmer_10.py
