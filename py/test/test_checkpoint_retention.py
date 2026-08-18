from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.experiment.configuration import load_chess_experiment_configuration
from src.games.representation import NetworkDimensions
from src.training.checkpoint import CheckpointManifest, CheckpointReference
from src.training.checkpoint.retention import CheckpointRetention
from src.training.network import GoPointPassPolicyHeadConfiguration, NetworkDefinition, NetworkParams


NETWORK_DEFINITION = NetworkDefinition(
    architecture=NetworkParams(num_layers=1, hidden_size=8, policy_head=GoPointPassPolicyHeadConfiguration()),
    dimensions=NetworkDimensions(channels=3, rows=3, columns=3, actions=10),
    auxiliary_heads=(),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checkpoint(run_path: Path, generation: int) -> None:
    model_path = run_path / f'model_{generation}.pt'
    optimizer_path = run_path / f'optimizer_{generation}.pt'
    inference_path = run_path / f'model_{generation}.jit.pt'
    model_path.write_bytes(f'model-{generation}'.encode())
    optimizer_path.write_bytes(f'optimizer-{generation}'.encode())
    inference_path.write_bytes(f'inference-{generation}'.encode())
    manifest = CheckpointManifest(
        generation=generation,
        network=NETWORK_DEFINITION,
        model_path=model_path.name,
        model_sha256=_sha256(model_path),
        optimizer_path=optimizer_path.name,
        optimizer_sha256=_sha256(optimizer_path),
        inference_model_path=inference_path.name,
        inference_model_sha256=_sha256(inference_path),
    )
    (run_path / f'checkpoint_{generation}.json').write_text(manifest.model_dump_json(), encoding='utf-8')


def _retention(run_path: Path) -> CheckpointRetention:
    experiment = load_chess_experiment_configuration(Path('test/configs/chess-experiment.yaml'))
    inference_retention = experiment.training.lifecycle.inference_retention.validated_copy(
        update={'recent_checkpoint_count': 3, 'milestone_interval': 5}
    )
    credit = experiment.training.lifecycle.credit.validated_copy(update={'retained_checkpoint_interval_generations': 3})
    lifecycle = experiment.training.lifecycle.validated_copy(
        update={
            'credit': credit.model_dump(mode='json'),
            'inference_retention': inference_retention.model_dump(mode='json'),
        }
    )
    return CheckpointRetention(run_path, lifecycle)


def test_checkpoint_retention_keeps_resumable_intervals_and_required_inference_models(tmp_path: Path) -> None:
    for generation in range(16):
        _write_checkpoint(tmp_path, generation)

    _retention(tmp_path).apply(active_generation=15, required_inference_generations=(7, 8))

    retained_models = tuple(generation for generation in range(16) if (tmp_path / f'model_{generation}.pt').is_file())
    retained_optimizers = tuple(
        generation for generation in range(16) if (tmp_path / f'optimizer_{generation}.pt').is_file()
    )
    retained_inference = tuple(
        generation for generation in range(16) if (tmp_path / f'model_{generation}.jit.pt').is_file()
    )

    assert retained_models == (0, 3, 6, 9, 12, 15)
    assert retained_optimizers == retained_models
    assert retained_inference == (0, 5, 7, 8, 10, 13, 14, 15)
    assert all((tmp_path / f'checkpoint_{generation}.json').is_file() for generation in range(16))


def test_inference_checkpoint_load_does_not_require_pruned_training_artifacts(tmp_path: Path) -> None:
    for generation in range(9):
        _write_checkpoint(tmp_path, generation)
    _retention(tmp_path).apply(active_generation=8, required_inference_generations=(7,))

    checkpoint = CheckpointReference.load_for_inference(tmp_path, 7)

    assert checkpoint.generation == 7
    assert checkpoint.inference_model_path.is_file()
    with pytest.raises(ValueError, match='artifact does not exist'):
        CheckpointReference.load(tmp_path, 7)
