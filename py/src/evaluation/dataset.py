from __future__ import annotations

from pathlib import Path
import time
from typing import TypeVar

import numpy as np
import torch

from src.evaluation.artifacts import dataset_manifest_path, load_evaluation_dataset
from src.evaluation.contracts import EvaluationDatasetManifest, FixedDatasetEvaluationJob, FixedDatasetEvaluationResult
from src.games.contracts import GameStateContract
from src.packed_planes import PackedPlanePayload, decode_packed_planes_into


PositionT = TypeVar('PositionT')


def evaluate_fixed_dataset(
    job: FixedDatasetEvaluationJob,
    state: GameStateContract[PositionT],
    dataset_path: Path,
    device_type: str,
    batch_size: int = 256,
) -> FixedDatasetEvaluationResult:
    if batch_size <= 0:
        raise ValueError('Evaluation dataset batch size must be positive.')
    started_at = time.monotonic()
    manifest = EvaluationDatasetManifest.model_validate_json(
        dataset_manifest_path(dataset_path).read_text(encoding='utf-8')
    )
    data = load_evaluation_dataset(dataset_path, manifest)
    device = torch.device('cpu') if device_type == 'cpu' else torch.device('cuda', job.device_id)
    model = torch.jit.load(str(job.candidate.inference_model_path), map_location=device)
    model.eval()
    correct = 0
    cross_entropy = 0.0
    with torch.inference_mode():
        for start in range(0, manifest.position_count, batch_size):
            batch = data[start : start + batch_size]
            packed_states = tuple(state.packed_plane_layout.value(bytes(row['packed_state'])) for row in batch)
            decoded = np.empty(
                (
                    len(batch),
                    state.representation.channels,
                    state.representation.rows,
                    state.representation.columns,
                ),
                dtype=np.float32,
            )
            decode_packed_planes_into(
                tuple(PackedPlanePayload(value.payload) for value in packed_states),
                state.packed_plane_layout,
                state.representation.binary_channels,
                state.representation.scalar_channels,
                decoded,
            )
            policy, _ = model(torch.from_numpy(decoded).to(device))
            policy = policy.float().cpu()
            top_actions = policy.argmax(dim=1).numpy()
            correct += int(np.count_nonzero(top_actions == batch['top_action_id']))
            for row_index, row in enumerate(batch):
                count = int(row['policy_count'])
                action_ids = torch.from_numpy(row['action_ids'][:count].astype(np.int64))
                targets = torch.from_numpy(row['probabilities'][:count].astype(np.float32))
                predicted = policy[row_index, action_ids].clamp_min(1e-12)
                cross_entropy -= float(torch.sum(targets * torch.log(predicted)).item())
    return FixedDatasetEvaluationResult(
        kind='fixed_dataset',
        job=job,
        position_count=manifest.position_count,
        source_game_count=len(manifest.source_games),
        top_action_accuracy=correct / manifest.position_count,
        policy_cross_entropy=cross_entropy / manifest.position_count,
        duration_seconds=time.monotonic() - started_at,
    )
