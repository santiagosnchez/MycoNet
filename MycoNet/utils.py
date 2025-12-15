"""Utility helpers for MycoNet training and runtime.

Small utilities that don't belong in `make_model` or `data`.
"""

import os
import logging
import tensorflow as tf


def checkpoint(path: str):
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Keras requires a .weights.h5 extension when save_weights_only=True
    if not path.endswith(".weights.h5"):
        path = path + ".weights.h5"
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=path, save_weights_only=True, verbose=1, save_freq="epoch"
    )


def configure_gpus():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            logging.info(
                "%d Physical GPUs, %d Logical GPUs", len(gpus), len(logical_gpus)
            )
        except RuntimeError as e:
            logging.warning("GPU configuration error: %s", e)
