import numpy as np
from MycoNet import recode


def test_flatten_DNA_all_bases():
    seq = "ACGTMRWSYKVHDBN"
    expected = [1, 2, 3, 4, 1, 1, 1, 2, 2, 3, 0, 0, 0, 0, 0]
    out = recode.flatten_DNA(seq)
    assert isinstance(out, list)
    assert out == expected


def test_flatten_labels_and_get_labels():
    labels = [0, 2]
    onehot = recode.flatten_labels(labels)
    assert onehot.shape == (2, 3)
    assert onehot.dtype == np.int8
    assert onehot[0, 0] == 1
    assert onehot[1, 2] == 1

    data = [("L1", "ACTG"), ("L2", "GGTT")]
    names = recode.get_labels(data, flatten=False)
    assert names == ["L1", "L2"]
    # when flatten=True the labels are expected to be numeric indices
    data_numeric = [(0, "ACTG"), (1, "GGTT")]
    labs = recode.get_labels(data_numeric, flatten=True)
    assert labs.shape[0] == 2


def test_get_recoded_sequences_non_kmer_padding():
    data = [("h1", "ACTG"), ("h2", "GGTT")]
    X, vocab = recode.get_recoded_sequences(data, pad=True, longest=6, kmer=0)
    assert vocab == 4
    assert X.shape == (2, 6)
    assert X.dtype == np.int8
    # ensure non-zero values (1..4) are present
    assert np.any(X > 0)
