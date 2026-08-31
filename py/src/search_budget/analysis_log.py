from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from src.search_budget.policy import BUDGET_CURVE_POINTS

ANALYSIS_RECORD_DTYPE = np.dtype(
    [
        ('source_generation', '<u4'),
        ('model_generation', '<u4'),
        ('ply', '<u4'),
        ('first_absolute_replay_row', '<u8'),
        ('baseline_visits', '<u4'),
        ('policy_kl', '<f4', (BUDGET_CURVE_POINTS,)),
        ('value_error', '<f4', (BUDGET_CURVE_POINTS,)),
        ('top_visit_share', '<f4'),
        ('policy_entropy', '<f4'),
        # Root-time approximations actually available at selection (raw-prior based on a fresh
        # root), logged beside the post-search values so the deployment gap stays measurable.
        ('root_prior_top_share', '<f4'),
        ('root_prior_entropy', '<f4'),
        ('predicted_curve', '<f4', (BUDGET_CURVE_POINTS,)),
        ('corrected_curve', '<f4', (BUDGET_CURVE_POINTS,)),
        ('deep_half_kl', '<f4'),
        ('assigned_visits', '<u4'),
        ('selected_index', '<u4'),
    ]
)


def analysis_log_path(labels_path: Path, source_generation: int) -> Path:
    return labels_path / f'analysis-generation-{source_generation:08d}.np'


def append_analysis_records(path: Path, records: npt.NDArray[np.void]) -> None:
    if records.dtype != ANALYSIS_RECORD_DTYPE:
        raise ValueError('Analysis records must use the fixed analysis record dtype.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('ab') as stream:
        stream.write(records.tobytes())


def read_analysis_records(path: Path) -> npt.NDArray[np.void]:
    payload = path.read_bytes()
    if len(payload) % ANALYSIS_RECORD_DTYPE.itemsize:
        raise ValueError('Analysis log length is not a whole number of fixed-width records.')
    return np.frombuffer(payload, dtype=ANALYSIS_RECORD_DTYPE)
