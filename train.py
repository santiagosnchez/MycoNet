"""Training entrypoint (promoted CLI).

This file is a promoted copy of `training_cli.py` and is the canonical
training entrypoint. The original legacy script was archived as
`training_legacy.py`.
"""

# Promoted from training_cli.py

import os
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from MycoNet.recode import get_recoded_sequences
from MycoNet.kmer_embedding import *
from MycoNet.make_model import make_model
from MycoNet.data import load_or_prepare_data, ensure_dirs
from MycoNet.utils import checkpoint, configure_gpus


def parse_args():
    p = argparse.ArgumentParser(description="Train MycoNet model on recoded sequences")
    p.add_argument(
        "fasta", help="Input FASTA-like file (header/sequence alternating lines)"
    )
    p.add_argument("name", help="Output experiment directory name")
    p.add_argument("kmer", type=int, help="k (or pass 0/False to use base recoding)")
    p.add_argument("--maxseq", type=int, default=1500)
    p.add_argument("--minseq", type=int, default=100)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--embedding-dim", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--lstm-dim", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    # GPU memory growth
    configure_gpus()

    ensure_dirs(args.name)
    X_train, X_test, y_train, y_test, vocab_size = load_or_prepare_data(
        args.fasta, args.name, args.kmer, args.maxseq, args.minseq, args.test_size
    )

    logging.info(
        "data sizes: X_train=%s, X_test=%s, y_train=%s, y_test=%s",
        X_train.shape,
        X_test.shape,
        y_train.shape,
        y_test.shape,
    )

    num_classes = int(np.max(np.concatenate([y_train, y_test])) + 1)
    model = make_model(
        input_dim=vocab_size + 1,
        output_layer=num_classes,
        dropout=args.dropout,
        embedding_dim=args.embedding_dim,
        activation=0.01,
        learning_rate=args.learning_rate,
        LSTM_dim=args.lstm_dim,
    )

    # Paths for saving/loading full model and checkpoint
    model_path_keras = os.path.join(
        args.name, "training", args.name + "_entire_model.keras"
    )
    model_path_saved = os.path.join(args.name, "training", args.name + "_entire_model")
    chk_path = os.path.join(args.name, "training", args.name + ".cp.ckpt")

    # Prefer loading the single-file Keras format if present; otherwise load weights from checkpoint
    if os.path.exists(model_path_keras):
        logging.info("loading saved model %s", model_path_keras)
        model = tf.keras.models.load_model(model_path_keras)
    elif os.path.exists(chk_path + ".index"):
        logging.info("checkpoint found, loading weights from %s", chk_path)
        model.load_weights(chk_path)

    cp_callback = checkpoint(chk_path)
    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[cp_callback],
    )

    # Save the entire model as a single-file `.keras` so `load_model` works with Keras 3.
    try:
        model.save(model_path_keras)
        logging.info("Saved full model to %s", model_path_keras)
    except Exception as e:
        logging.warning("Failed to save .keras model: %s", e)

    # Also attempt to export a SavedModel directory if `model.export` is available.
    try:
        if hasattr(model, "export"):
            model.export(model_path_saved)
            logging.info("Exported SavedModel to %s", model_path_saved)
        else:
            logging.info("model.export not available; skipping SavedModel export")
    except Exception as e:
        logging.warning("SavedModel export failed: %s", e)

    # save history
    hist_df = pd.DataFrame.from_dict(history.history)
    hist_path = os.path.join(args.name, "training", "history.csv")
    hist_df.to_csv(hist_path, mode="a", header=False, index=False)


if __name__ == "__main__":
    main()
