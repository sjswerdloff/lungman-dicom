"""Where the Lungman archive is, for the tests that need the real data.

`LUNGMAN_DATA` names the archive directory (the one holding `CD1/` and `SEGMENTATION/`). Without it the tests
look in `~/Downloads/lungman_data` and skip when it is absent. With it set, a missing archive stops the whole
run (see conftest.py): a run that was told where the data is must not pass by skipping.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ENV_VAR = "LUNGMAN_DATA"
LUNGMAN = Path(os.environ.get(ENV_VAR) or Path.home() / "Downloads" / "lungman_data")
REQUIRED = bool(os.environ.get(ENV_VAR))
AVAILABLE = (LUNGMAN / "SEGMENTATION" / "labels.dat").is_file()

needs_lungman = pytest.mark.skipif(
    not AVAILABLE, reason=f"Lungman data not found at {LUNGMAN} (set {ENV_VAR} to its location)"
)
