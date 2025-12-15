import os

import pytest
import tensorflow as tf

from MycoNet import utils as mutils


def test_checkpoint_returns_callback(tmp_path):
    path = tmp_path / "ckpt" / "model.ckpt"
    cb = mutils.checkpoint(str(path))
    # should be a keras callback (ModelCheckpoint)
    assert isinstance(cb, tf.keras.callbacks.Callback)


def test_configure_gpus_no_throw():
    # Should not raise even if there are no GPUs in CI
    mutils.configure_gpus()
