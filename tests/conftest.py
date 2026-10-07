"""Stop the run when the Lungman data was asked for and is not there."""

from __future__ import annotations

import pytest
from lungman_fixture import AVAILABLE, ENV_VAR, LUNGMAN, REQUIRED


def pytest_sessionstart(session: pytest.Session) -> None:
    if REQUIRED and not AVAILABLE:
        pytest.exit(f"{ENV_VAR} is set to {LUNGMAN}, but {LUNGMAN / 'SEGMENTATION' / 'labels.dat'} is not there", returncode=2)
