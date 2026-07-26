from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import uuid

import h5py
import numpy as np
import numpy.typing as npt


REANALYSIS_SIDECAR_SCHEMA_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReanalysisPosition:
    row_index: int
    starting_fen: str
    moves_uci: tuple[str, ...]


@dataclass(frozen=True)
class ReanalysisTarget:
    row_index: int
    visit_counts: npt.NDArray[np.uint16]
    mcts_root_value: float


def write_reanalysis_sidecar(
    source_payload: Path,
    model_version: int,
    targets: tuple[ReanalysisTarget, ...],
) -> Path:
    if model_version < 0 or not targets:
        raise ValueError('Reanalysis sidecars require a model version and at least one target.')
    maximum_visits = max(len(target.visit_counts) for target in targets)
    padded_visits = np.zeros((len(targets), maximum_visits, 2), dtype=np.int32)
    for target_index, target in enumerate(targets):
        padded_visits[target_index, : len(target.visit_counts)] = target.visit_counts
    path = source_payload.with_name(f'{source_payload.stem}.reanalysis-{model_version:010d}.hdf5')
    temporary_path = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with h5py.File(temporary_path, 'w') as file:
        file.create_dataset(
            'row_indices',
            data=np.fromiter((target.row_index for target in targets), dtype=np.int64),
        )
        file.create_dataset('visit_counts', data=padded_visits)
        file.create_dataset(
            'mcts_root_values',
            data=np.fromiter((target.mcts_root_value for target in targets), dtype=np.float32),
        )
        file.attrs['reanalysis_schema_version'] = REANALYSIS_SIDECAR_SCHEMA_VERSION
        file.attrs['source_payload_name'] = source_payload.name
        file.attrs['source_payload_sha256'] = _file_sha256(source_payload)
        file.attrs['model_version'] = model_version
    os.replace(temporary_path, path)
    return path


def latest_reanalysis_overrides(
    source_payload: Path,
    expected_source_sha256: str,
) -> dict[int, tuple[npt.NDArray[np.uint16], float]]:
    sidecars = sorted(source_payload.parent.glob(f'{source_payload.stem}.reanalysis-*.hdf5'))
    overrides: dict[int, tuple[npt.NDArray[np.uint16], float]] = {}
    for sidecar in sidecars:
        with h5py.File(sidecar, 'r') as file:
            if int(file.attrs['reanalysis_schema_version']) != REANALYSIS_SIDECAR_SCHEMA_VERSION:
                raise ValueError(f'Unsupported reanalysis sidecar schema: {sidecar}')
            if str(file.attrs['source_payload_sha256']) != expected_source_sha256:
                raise ValueError(f'Reanalysis sidecar source identity mismatch: {sidecar}')
            row_indices = np.asarray(file['row_indices'][...], dtype=np.int64)
            visits = np.asarray(file['visit_counts'][...], dtype=np.int32)
            root_values = np.asarray(file['mcts_root_values'][...], dtype=np.float32)
        overrides.update(
            {
                int(row_index): (
                    row_visits[row_visits[:, 1] > 0].astype(np.uint16, copy=False),
                    float(root_value),
                )
                for row_index, row_visits, root_value in zip(row_indices, visits, root_values)
            }
        )
    return overrides
