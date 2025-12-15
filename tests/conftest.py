import sys
from pathlib import Path

# Ensure repo root is on sys.path so tests can import the MycoNet package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
