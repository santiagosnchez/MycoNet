import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_predict_file_based(tmp_path):
    # Create a workspace directory so predict.py can receive a relative model_dir
    work = tmp_path / "work"
    work.mkdir()

    exp = work / "exp"
    training = exp / "training"
    inputs = exp / "inputs"
    training.mkdir(parents=True)
    inputs.mkdir(parents=True)

    # Build and save a tiny model as a SavedModel directory under training/exp_entire_model
    from MycoNet.make_model import make_model

    model = make_model(2, 4, 2, 0.1, 0.01, 0.01, 4)
    model_path = training / "exp_entire_model"
    # save as SavedModel directory (use `export` when available for Keras3)
    if hasattr(model, "export"):
        # build the model for the expected input length (maxseq - kmer)
        maxseq = 1500
        kmer = 10
        input_len = maxseq - kmer
        try:
            model.build((None, input_len))
        except Exception:
            # fallback: run a single predict to build the model
            import numpy as _np

            model.predict(_np.zeros((1, input_len), dtype=_np.int32))
        model.export(str(model_path))
    else:
        # fallback: save as a .keras archive (predict.py expects a SavedModel dir,
        # but older TF may accept a .keras file if named without extension elsewhere)
        model.save(str(model_path) + ".keras")

    # labels_key_dict used by predict.py
    with open(inputs / "labels_key_dict.txt", "w") as f:
        f.write("0\tA\n1\tB\n")

    # simple kmer_dict (predict.py expects this file)
    with open(inputs / "kmer_dict.txt", "w") as f:
        f.write("AAA\t1\n")

    # create a small fasta with sequences long enough to pass filtering (>=100)
    seq1 = "ACGT" * 38  # 152 bases
    seq2 = "GCTA" * 30  # 120 bases
    fasta = work / "sample.fasta"
    fasta.write_text(f">h1|A\n{seq1}\n>h2|B\n{seq2}\n")

    outcsv = work / "out.csv"

    # Run the top-level predict.py script in the work directory so model_dir is 'exp'
    predict_py = Path(__file__).resolve().parents[1] / "predict.py"
    cmd = [sys.executable, str(predict_py), str(fasta.name), "exp", str(outcsv.name)]
    subprocess.run(cmd, cwd=str(work), check=True)

    assert outcsv.exists()
    df = pd.read_csv(outcsv)
    assert set(["SH", "prob", "org_label"]).issubset(df.columns)
