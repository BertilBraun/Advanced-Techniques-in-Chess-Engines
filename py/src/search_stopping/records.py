from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from src.search_stopping.features import STOP_PREDICTOR_FEATURE_COUNT

PAIRED_FLOOR_RECORD_DTYPE = np.dtype(
    [
        ('source_generation', '<u4'),
        ('ply', '<u4'),
        ('baseline_visits', '<u4'),
        ('kl_symmetric', '<f4'),
        ('value_gap', '<f4'),
    ]
)

ANCHOR_RECORD_DTYPE = np.dtype(
    [
        ('source_generation', '<u4'),
        ('ply', '<u4'),
        ('baseline_visits', '<u4'),
        ('kl_anchor_to_capped', '<f4'),
    ]
)


def audit_record_dtype(checkpoint_count: int) -> np.dtype:
    """Raw audit evidence: labels are derived at solve time from the stored KLs and gaps, never
    baked in, so eps can move without invalidating the window."""
    if checkpoint_count <= 0:
        raise ValueError('Audit records require at least one checkpoint.')
    return np.dtype(
        [
            ('source_generation', '<u4'),
            ('model_generation', '<u4'),
            ('game_key', '<u8'),
            ('ply', '<u4'),
            ('baseline_visits', '<u4'),
            ('starting_visits', '<u4'),
            ('final_visits', '<u4'),
            ('final_root_value', '<f4'),
            ('kl_to_final', '<f4', (checkpoint_count,)),
            ('value_gap', '<f4', (checkpoint_count,)),
            ('argmax_swap', '<u1', (checkpoint_count,)),
            ('guard_movement', '<f4', (checkpoint_count,)),
            ('stop_probability', '<f4', (checkpoint_count,)),
            ('would_stop', '<u1', (checkpoint_count,)),
            ('features', '<f4', (checkpoint_count, STOP_PREDICTOR_FEATURE_COUNT)),
        ]
    )


def audit_log_path(stopping_path: Path, source_generation: int, worker_id: int) -> Path:
    return stopping_path / f'audit-generation-{source_generation:08d}-worker-{worker_id:04d}.np'


def paired_floor_log_path(stopping_path: Path, source_generation: int, worker_id: int) -> Path:
    return stopping_path / f'paired-floor-generation-{source_generation:08d}-worker-{worker_id:04d}.np'


def anchor_log_path(stopping_path: Path, source_generation: int, worker_id: int) -> Path:
    return stopping_path / f'anchor-generation-{source_generation:08d}-worker-{worker_id:04d}.np'


def append_records(path: Path, records: npt.NDArray[np.void], dtype: np.dtype) -> None:
    if records.dtype != dtype:
        raise ValueError('Records must use the expected fixed-width dtype.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('ab') as stream:
        stream.write(records.tobytes())


def read_records(path: Path, dtype: np.dtype) -> npt.NDArray[np.void]:
    payload = path.read_bytes()
    if len(payload) % dtype.itemsize:
        raise ValueError('Record log length is not a whole number of fixed-width records.')
    return np.frombuffer(payload, dtype=dtype)
