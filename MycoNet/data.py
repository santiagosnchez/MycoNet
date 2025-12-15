"""Data preprocessing utilities for MycoNet.

Contains functions to read input files, preprocess sequences and labels,
and prepare train/test splits. These are imported by the CLI training
script to keep top-level code compact and testable.
"""

from typing import List, Tuple
import os
import logging
import numpy as np

from MycoNet.recode import get_recoded_sequences
from MycoNet.make_model import split_train_test


def split_data(fasta: str) -> List[Tuple[str, str]]:
    with open(fasta) as f:
        data = f.read().splitlines()
    idx = iter(range(len(data)))
    return [[data[i], data[j].upper()] for i, j in zip(idx, idx)]


def preprocessdata(fasta: str, maxlen: int, minlen: int):
    logging.info("splitting data")
    data = split_data(fasta)
    seqlens = np.array([len(s) for h, s in data])
    logging.info("max sequence length: %d", seqlens.max())
    logging.info("min sequence length: %d", seqlens.min())
    logging.info("filtering; keep sequences with len <= %d and >= %d", maxlen, minlen)
    data = [[x, y] for x, y in data if len(y) <= maxlen and len(y) >= minlen]
    return data


def preprocesslabels(data, name: str):
    logging.info("processing labels")
    labels_short = [x[0].split("|")[-1] for x in data]
    labels_unique = list(set(labels_short))
    labels_dict = {y: x for x, y in enumerate(labels_unique)}
    labels_dict_back = {x: y for x, y in enumerate(labels_unique)}
    os.makedirs(os.path.join(name, "inputs"), exist_ok=True)
    with open(os.path.join(name, "inputs", "labels_key_dict.txt"), "w") as o:
        for i in labels_dict_back.keys():
            o.write(str(i) + "\t" + labels_dict_back[i] + "\n")
    labels_numb = np.array([labels_dict[x] for x in labels_short], dtype=np.int32)
    return labels_numb


def ensure_dirs(name: str):
    os.makedirs(name, exist_ok=True)
    os.makedirs(os.path.join(name, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(name, "training"), exist_ok=True)


def load_or_prepare_data(
    fasta_db: str, name: str, kmer: int, maxseq: int, minseq: int, test_size: float
):
    """Return X_train, X_test, y_train, y_test, vocab_size.

    This function will reuse saved numpy files under `name/inputs/` if present,
    otherwise it will preprocess sequences and save intermediate files.
    """
    inputs_dir = os.path.join(name, "inputs")
    # load pre-saved train/test if present
    xt_path = os.path.join(inputs_dir, "X_train.npy")
    if os.path.exists(xt_path):
        logging.info("loading saved train/test sets")
        X_train = np.load(os.path.join(inputs_dir, "X_train.npy"))
        y_train = np.load(os.path.join(inputs_dir, "y_train.npy"))
        X_test = np.load(os.path.join(inputs_dir, "X_test.npy"))
        y_test = np.load(os.path.join(inputs_dir, "y_test.npy"))
        with open(os.path.join(inputs_dir, "vocab_size.txt"), "r") as f:
            vocab_size = int(f.read().strip())
        return X_train, X_test, y_train, y_test, vocab_size

    data_matrix_path = os.path.join(inputs_dir, "data_sequence_matrix.npy")
    labels_matrix_path = os.path.join(inputs_dir, "labels_matrix.npy")
    if os.path.exists(data_matrix_path) and os.path.exists(labels_matrix_path):
        logging.info("loading data matrix")
        X = np.load(data_matrix_path)
        labels_idx = np.load(labels_matrix_path)
        with open(os.path.join(inputs_dir, "vocab_size.txt"), "r") as f:
            vocab_size = int(f.read().strip())
    else:
        data = preprocessdata(fasta_db, maxseq, minseq)
        labels_idx = preprocesslabels(data, name)
        np.save(labels_matrix_path, labels_idx)
        logging.info("recoding DNA and padding")
        X, vocab_size = get_recoded_sequences(data, pad=True, longest=maxseq, kmer=kmer)
        logging.info("final X dim: %s", str(X.shape))
        np.save(data_matrix_path, X)
        with open(os.path.join(inputs_dir, "vocab_size.txt"), "w") as o:
            o.write(str(vocab_size) + "\n")

    X_train, X_test, y_train, y_test = split_train_test(
        X, labels_idx, test_size=test_size
    )
    np.save(os.path.join(inputs_dir, "X_train.npy"), X_train)
    np.save(os.path.join(inputs_dir, "X_test.npy"), X_test)
    np.save(os.path.join(inputs_dir, "y_train.npy"), y_train)
    np.save(os.path.join(inputs_dir, "y_test.npy"), y_test)
    return X_train, X_test, y_train, y_test, vocab_size
