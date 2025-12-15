import sys
from pathlib import Path
import subprocess
import shutil

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


@pytest.fixture(scope="session", autouse=True)
def _cleanup_small_tmp_dirs(tmp_path_factory):
    """Ensure any `small_test*` dirs created under pytest basetemp are removed.

    Some tests or external tools may create temporary directories named
    `small_test*` under the pytest base temp. This autouse fixture removes
    them at session teardown to avoid leftover clutter.
    """
    yield
    try:
        base = Path(tmp_path_factory.getbasetemp())
        for p in base.iterdir():
            if p.name.startswith("small_test"):
                shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass
