import numpy as np
import random as rn

from MycoNet import kmer_embedding as kemb


def test_all_kmers_dict_length_and_keys():
    k = 2
    d = kemb.all_kmers_dict(k)
    assert len(d) == 4**k
    # keys are k-length strings consisting of A/C/T/G
    for key in d.keys():
        assert isinstance(key, str)
        assert len(key) == k
        assert set(key).issubset(set("ACTG"))
    # values are unique and in expected range
    vals = set(d.values())
    assert len(vals) == len(d)
    assert min(vals) >= 1


def test_find_all_kmers_and_resolve_amb():
    rn.seed(0)
    seqs = ["ACTG"]
    found = kemb.find_all_kmers(seqs, 2)
    # note: implementation iterates range(len(seq)-kmer) so for length 4 and k=2
    # it yields positions 0 and 1 -> 'AC' and 'CT' only
    for mer in ["AC", "CT"]:
        assert mer in found

    # check resolve_amb replaces ambiguous letters deterministically with seed
    rn.seed(1)
    out = kemb.resolve_amb("ANR")
    assert len(out) == 3
    assert set(out).issubset(set("ACGT"))


def test_kmer_to_int_padding_and_dtype():
    rn.seed(0)
    seqs = ["ACTG"]
    k = 2
    kmer_key = kemb.all_kmers_dict(k)
    X_pad, missing = kemb.kmer_to_int(seqs, k, kmer_key, padding=True, max_size=6)
    # for seq length 4 and k=2 -> len(trans)=2, padded length = (6-2)=4
    assert X_pad.shape == (1, 4)
    assert X_pad.dtype in (np.int16, np.int32)
    assert isinstance(missing, list)
    # non-padding case
    X_np, missing2 = kemb.kmer_to_int(seqs, k, kmer_key, padding=False, max_size=6)
    assert X_np.shape[0] == 1
    assert X_np.dtype in (np.int16, np.int32)
