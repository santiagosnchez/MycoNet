"""Utilities package for MycoNet.

Keep a small, stable public surface so callers can `from MycoNet.utils
import checkpoint, configure_gpus` without needing to track internal
simulation helpers. Simulation helpers live in
`MycoNet.utils.simulate_dna_data` and should be imported explicitly when
needed.
"""

from .utils import checkpoint, configure_gpus

__all__ = ["checkpoint", "configure_gpus"]
