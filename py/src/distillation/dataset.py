from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from pydantic import Field
from src.games.contracts import GameStateContract
from src.games.representation import decode_packed_plane_bytes_into
from src.training.batch import TrainingBatch
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel

MAXIMUM_POLICY_ENTRIES = 64
MAXIMUM_LEGAL_ACTIONS = 218
CHESS_PAYLOAD_BYTES = 183


class DistillationDatasetManifest(FrozenModel):
    # Version 2 replaced the fixed ply stride with independent per-ply retention; version 1 files do not load.
    schema_version: int = Field(default=2, ge=2)
    game: str = Field(min_length=1)
    position_count: int = Field(gt=0)
    action_size: int = Field(gt=0)
    payload_bytes: int = Field(gt=0)
    maximum_policy_entries: int = Field(gt=0, le=255)
    maximum_legal_actions: int = Field(gt=0, le=255)
    teacher_generation: int = Field(ge=0)
    teacher_weights_sha256: str = Field(min_length=64, max_length=64)
    teacher_parameter_count: int = Field(gt=0)
    random_seed: int = Field(ge=0)
    random_opening_plies: int = Field(ge=0)
    sampling_temperature: float = Field(gt=0.0)
    sample_one_position_in: int = Field(gt=0)
    random_perturbation_probability: float = Field(ge=0.0, le=1.0)
    maximum_game_plies: int = Field(gt=0)
    builder_source_revision: str = Field(min_length=1)
    # Present only on merged datasets; the generator settings above then describe the last source, whose rows form
    # the held-out tail.
    merged_sources: tuple[str, ...] = ()


def record_dtype(payload_bytes: int = CHESS_PAYLOAD_BYTES) -> np.dtype:
    return np.dtype(
        [
            ('packed_state', f'V{payload_bytes}'),
            ('legal_count', '<u2'),
            ('legal_action_ids', '<u2', (MAXIMUM_LEGAL_ACTIONS,)),
            ('policy_count', '<u2'),
            ('policy_action_ids', '<u2', (MAXIMUM_POLICY_ENTRIES,)),
            ('policy_probabilities', '<f4', (MAXIMUM_POLICY_ENTRIES,)),
            ('wdl', '<f4', (3,)),
        ]
    )


def manifest_path(dataset_path: Path) -> Path:
    return dataset_path.with_suffix('.manifest.json')


def write_dataset(dataset_path: Path, records: npt.NDArray, manifest: DistillationDatasetManifest) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    records.tofile(dataset_path)
    write_text_atomically(manifest_path(dataset_path), manifest.model_dump_json(indent=2))


def read_manifest(dataset_path: Path) -> DistillationDatasetManifest:
    return DistillationDatasetManifest.model_validate_json(manifest_path(dataset_path).read_text(encoding='utf-8'))


def open_dataset(dataset_path: Path) -> tuple[npt.NDArray, DistillationDatasetManifest]:
    manifest = read_manifest(dataset_path)
    records = np.memmap(dataset_path, dtype=record_dtype(manifest.payload_bytes), mode='r')
    if len(records) != manifest.position_count:
        raise ValueError(f'Dataset holds {len(records)} rows but its manifest declares {manifest.position_count}.')
    return records, manifest


def build_training_batch(
    rows: npt.NDArray,
    state: GameStateContract,
    action_size: int,
    device: torch.device,
) -> TrainingBatch:
    batch_size = len(rows)

    # The record itemsize is odd, so a single-row slice keeps a stride torch.from_numpy rejects; copy unconditionally.
    packed = np.array(rows['packed_state']).view(np.uint8).reshape(batch_size, -1)
    decoded = np.empty(
        (batch_size, state.representation.channels, state.representation.rows, state.representation.columns),
        dtype=np.float32,
    )
    decode_packed_plane_bytes_into(
        packed,
        state.packed_plane_layout,
        state.representation.binary_channels,
        state.representation.scalar_channels,
        decoded,
    )

    policy_targets = np.zeros((batch_size, action_size), dtype=np.float32)
    entry_columns = np.arange(MAXIMUM_POLICY_ENTRIES)
    entry_mask = entry_columns[None, :] < rows['policy_count'][:, None]
    row_indices = np.repeat(np.arange(batch_size), entry_mask.sum(axis=1))
    policy_targets[row_indices, rows['policy_action_ids'][entry_mask]] = rows['policy_probabilities'][entry_mask]

    legal_action_ids = np.full((batch_size, MAXIMUM_LEGAL_ACTIONS), -1, dtype=np.int64)
    legal_mask = np.arange(MAXIMUM_LEGAL_ACTIONS)[None, :] < rows['legal_count'][:, None]
    legal_action_ids[legal_mask] = rows['legal_action_ids'][legal_mask]

    return TrainingBatch(
        states=torch.from_numpy(decoded).to(device=device, non_blocking=True),
        policy_targets=torch.from_numpy(policy_targets).to(device=device, non_blocking=True),
        policy_legal_action_ids=torch.from_numpy(legal_action_ids).to(device=device, non_blocking=True),
        wdl_targets=torch.from_numpy(np.array(rows['wdl'], dtype=np.float32)).to(device=device, non_blocking=True),
        root_values=torch.zeros(batch_size, device=device),
        auxiliary_targets=(),
        auxiliary_legal_action_ids=(),
        auxiliary_eligibility=(),
        sample_weights=torch.ones(batch_size, device=device),
        source_model_generations=torch.zeros(batch_size, dtype=torch.int64, device=device),
        source_created_at_seconds=torch.zeros(batch_size, dtype=torch.float64, device=device),
    )
