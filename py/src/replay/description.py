from __future__ import annotations

from pathlib import Path

from src.replay.layout import ReplayLayout
from src.util.frozen_model import FrozenModel


class ReplayDescription(FrozenModel):
    path: Path
    head: int
    size: int
    logical_capacity: int
    maximum_capacity: int
    layout: ReplayLayout
