import sys
from pathlib import Path
import subprocess

import pytest

# Ensure repo root is on sys.path so tests can import the MycoNet package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory):
    """Run a tiny training job to produce a model directory for integration tests.

    Returns the path to the experiment directory (string).
    """
    tmp = tmp_path_factory.mktemp("trained")
    fasta = tmp / "sample.fasta"
    fasta.write_text(
        ">org1\nACGTACGTAC\n>org2\nACGTACGTAC\n>org3\nACGTACGTAC\n>org4\nACGTACGTAC\n"
    )

    expdir = tmp / "exp"
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        str(fasta),
        str(expdir),
        "0",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--maxseq",
        "50",
        "--minseq",
        "1",
        "--embedding-dim",
        "8",
        "--lstm-dim",
        "4",
        "--learning-rate",
        "0.01",
    ]
    subprocess.run(cmd, check=True)
    return str(expdir)
