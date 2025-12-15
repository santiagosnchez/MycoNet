import os
import numpy as np
from pathlib import Path

import pytest

from MycoNet import data as mdata


def test_split_data_and_preprocesslabels(tmp_path):
    sample = tmp_path / "sample.fasta"
    sample.write_text(">h1|L1\nACTG\n>h2|L2\nGGTT\n")

    pairs = mdata.split_data(str(sample))
    assert len(pairs) == 2
    assert pairs[0][0].startswith(">h1")
    assert pairs[0][1] == "ACTG"

    # test preprocesslabels writes labels_key_dict
    outdir = tmp_path / "exp"
    outdir.mkdir()
    labels = mdata.preprocesslabels(pairs, str(outdir))
    assert labels.shape[0] == 2
    lk = outdir / "inputs" / "labels_key_dict.txt"
    assert lk.exists()
    text = lk.read_text()
    assert "L1" in text and "L2" in text


def test_load_or_prepare_data_with_existing_matrices(tmp_path):
    exp = tmp_path / "exp"
    inputs = exp / "inputs"
    inputs.mkdir(parents=True)

    # create dummy X and labels saved files
    X = np.zeros((3, 10), dtype=np.int16)
    labels = np.array([0, 1, 2], dtype=np.int32)
    np.save(inputs / "data_sequence_matrix.npy", X)
    np.save(inputs / "labels_matrix.npy", labels)
    (inputs / "vocab_size.txt").write_text("100\n")

    X_train, X_test, y_train, y_test, vocab_size = mdata.load_or_prepare_data(
        str(sample := tmp_path / "unused.fasta"),
        str(exp),
        kmer=4,
        maxseq=1500,
        minseq=10,
        test_size=0.5,
    )
    assert vocab_size == 100
    assert X_train.shape[1] == 10
    assert y_train.ndim == 1
