import os
from types import SimpleNamespace

import numpy as np

import pytest


def test_train_creates_artifacts(tmp_path, monkeypatch):
    # write tiny fasta (header, sequence alternating lines)
    fasta = tmp_path / "small.fasta"
    fasta.write_text(
        ">s1|A\nACGTACGTAC\n>s2|B\nACGTACGTAC\n>s3|A\nACGTACGTAC\n>s4|B\nACGTACGTAC\n"
    )

    exp = tmp_path / "exp"

    # prepare args namespace to bypass argparse
    args = SimpleNamespace(
        fasta=str(fasta),
        name=str(exp),
        kmer=0,
        maxseq=50,
        minseq=1,
        epochs=1,
        batch_size=2,
        test_size=0.5,
        dropout=0.1,
        embedding_dim=8,
        learning_rate=0.01,
        lstm_dim=4,
    )

    # monkeypatch parse_args in train module
    import train

    monkeypatch.setattr(train, "parse_args", lambda: args)

    # run training (should be quick with epochs=1)
    train.main()

    # check artifacts
    training_dir = exp / "training"
    inputs_dir = exp / "inputs"

    assert training_dir.exists()
    assert inputs_dir.exists()

    # expect inputs arrays saved
    assert (inputs_dir / "X_train.npy").exists()
    assert (inputs_dir / "y_train.npy").exists()

    # labels mapping must exist
    assert (inputs_dir / "labels_key_dict.txt").exists()
