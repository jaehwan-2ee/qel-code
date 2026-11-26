"""The baselines of the comparison: SDDR (sddr*.py) and the eight DFL methods run by OpenPTO (dfl*.py)."""

import sys
from pathlib import Path

# OpenPTO is vendored in third_party/predictiveco and imports itself as `openpto`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/predictiveco"))
