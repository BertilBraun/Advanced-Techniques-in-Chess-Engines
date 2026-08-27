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
    # Auxiliary head outputs are always captured because generation cannot be repeated cheaply; whether a student
    # trains on them is decided later. Order matches the teacher's own auxiliary head layout.
    captured_auxiliary_heads: tuple[str, ...] = ()
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
            ('next_policy_count', '<u2'),
            ('next_policy_action_ids', '<u2', (MAXIMUM_POLICY_ENTRIES,)),
            ('next_policy_probabilities', '<f4', (MAXIMUM_POLICY_ENTRIES,)),
            ('remaining_game_length', '<f4'),
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


def _dense_policy(
    rows: npt.NDArray, action_size: int, count_field: str, ids_field: str, probs_field: str
) -> np.ndarray:
    batch_size = len(rows)
    dense = np.zeros((batch_size, action_size), dtype=np.float32)
    entry_mask = np.arange(MAXIMUM_POLICY_ENTRIES)[None, :] < rows[count_field][:, None]
    row_indices = np.repeat(np.arange(batch_size), entry_mask.sum(axis=1))
    dense[row_indices, rows[ids_field][entry_mask]] = rows[probs_field][entry_mask]
    return dense


def build_training_batch(
    rows: npt.NDArray,
    state: GameStateContract,
    action_size: int,
    device: torch.device,
    auxiliary_heads: tuple[str, ...] = (),
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

    policy_targets = _dense_policy(rows, action_size, 'policy_count', 'policy_action_ids', 'policy_probabilities')

    legal_action_ids = np.full((batch_size, MAXIMUM_LEGAL_ACTIONS), -1, dtype=np.int64)
    legal_mask = np.arange(MAXIMUM_LEGAL_ACTIONS)[None, :] < rows['legal_count'][:, None]
    legal_action_ids[legal_mask] = rows['legal_action_ids'][legal_mask]
    legal_action_tensor = torch.from_numpy(legal_action_ids).to(device=device, non_blocking=True)
    scalar_legal_filler = torch.full((batch_size, MAXIMUM_LEGAL_ACTIONS), -1, dtype=torch.int64, device=device)
    eligible = torch.ones(batch_size, dtype=torch.bool, device=device)

    auxiliary_targets: list[torch.Tensor] = []
    auxiliary_legal_action_ids: list[torch.Tensor] = []
    for head in auxiliary_heads:
        match head:
            case 'next_policy':
                dense = _dense_policy(
                    rows, action_size, 'next_policy_count', 'next_policy_action_ids', 'next_policy_probabilities'
                )
                auxiliary_targets.append(torch.from_numpy(dense).to(device=device, non_blocking=True))
                auxiliary_legal_action_ids.append(legal_action_tensor)
            case 'remaining_game_length':
                scalar = np.array(rows['remaining_game_length'], dtype=np.float32).reshape(batch_size, 1)
                auxiliary_targets.append(torch.from_numpy(scalar).to(device=device, non_blocking=True))
                auxiliary_legal_action_ids.append(scalar_legal_filler)
            case _:
                raise ValueError(f'Distillation datasets do not carry an auxiliary head named {head!r}.')

    return TrainingBatch(
        states=torch.from_numpy(decoded).to(device=device, non_blocking=True),
        policy_targets=torch.from_numpy(policy_targets).to(device=device, non_blocking=True),
        policy_legal_action_ids=legal_action_tensor,
        wdl_targets=torch.from_numpy(np.array(rows['wdl'], dtype=np.float32)).to(device=device, non_blocking=True),
        root_values=torch.zeros(batch_size, device=device),
        auxiliary_targets=tuple(auxiliary_targets),
        auxiliary_legal_action_ids=tuple(auxiliary_legal_action_ids),
        auxiliary_eligibility=tuple(eligible for _ in auxiliary_heads),
        sample_weights=torch.ones(batch_size, device=device),
        source_model_generations=torch.zeros(batch_size, dtype=torch.int64, device=device),
        source_created_at_seconds=torch.zeros(batch_size, dtype=torch.float64, device=device),
    )
