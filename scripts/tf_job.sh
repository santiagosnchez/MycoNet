#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00

set -euo pipefail

# Prefer a local virtualenv if present, otherwise try to load conda env
if [ -d ".venv" ]; then
	# shellcheck source=/dev/null
	source .venv/bin/activate
else
	if command -v module >/dev/null 2>&1; then
		module load anaconda3 || true
	fi
	if command -v conda >/dev/null 2>&1; then
		conda activate tf_gpu2.1 || true
	fi
fi

# Defaults (can be overridden by positional args)
FASTA=${1:-ITSx/SH_at_least_5_seqs.ITSx.concat_nogap.fasta}
EXP=${2:-kmer_7}
KMER=${3:-7}
EPOCHS=${4:-50}
BATCH=${5:-256}
MAXSEQ=${6:-1500}
MINSSEQ=${7:-100}

echo "Running training: fasta=$FASTA exp=$EXP kmer=$KMER epochs=$EPOCHS batch=$BATCH"

python -B train.py "$FASTA" "$EXP" "$KMER" \
	--epochs "$EPOCHS" --batch-size "$BATCH" --maxseq "$MAXSEQ" --minseq "$MINSSEQ"

echo "Training finished for experiment $EXP"
