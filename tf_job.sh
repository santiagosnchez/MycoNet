#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00

module load anaconda3
source activate tf_gpu2.1
python -B training.py ITSx/SH_at_least_5_seqs.ITSx.concat_nogap.fasta kmer_7 7
