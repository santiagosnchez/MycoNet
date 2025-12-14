#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
#SBATCH --time=12:00:00

ITSx -i SH_at_least_5_seqs.fa -o SH_at_least_5_seqs.ITSx --partial 200 --concat T --preserve T --cpu 80 -t F -p ~/software/ITSx_1.1.2/ITSx_db/HMMs/