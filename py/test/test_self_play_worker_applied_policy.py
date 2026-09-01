"""End-to-end regression for the v21 crash-loop: a real self-play worker batch must reach the
real native search under an APPLIED stop policy, with audit and non-audit positions mixed.
Every earlier gate was green while this path was broken, because nothing drove
SelfPlayWorker.run_batch against the native request validator with apply_learned=True."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip('AlphaZeroCpp')
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.training import ChessImplementation
from src.search_stopping.configuration import SearchStoppingConfiguration
from src.search_stopping.features import STOP_PREDICTOR_FEATURE_COUNT
from src.search_stopping.policy import SearchStopPolicy, closed_policy
from src.search_stopping.predictor import export_stop_predictor, fit_stop_predictor
from src.search_stopping.records import audit_record_dtype, read_records
from src.self_play.worker import SelfPlayWorker
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.paths import checkpoint_manifest_path
from test_helpers.configuration_paths import TEST_CONFIG_DIRECTORY

_BASELINE_VISITS = 24
_GAME_COUNT = 12


class _UniformChessModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = inputs.size(0)
        policy = torch.zeros((batch, 1880), dtype=inputs.dtype, device=inputs.device)
        outcome = torch.tensor([0.4, 0.3, 0.3], dtype=inputs.dtype, device=inputs.device)
        return policy, outcome.repeat(batch, 1)


def _configuration() -> ChessExperimentConfiguration:
    payload = yaml.safe_load((TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml').read_text(encoding='utf-8'))
    stopping = payload['training']['lifecycle']['search_stopping']
    stopping['audit_sample_fraction'] = '0.5'
    stopping['paired_audit_fraction'] = '0.5'
    stopping['anchor_fraction'] = '0.5'
    payload['chess']['self_play']['search']['baseline_visits'] = _BASELINE_VISITS
    return ChessExperimentConfiguration.model_validate(payload)


def _checkpoint(directory: Path, generation: int) -> CheckpointReference:
    directory.mkdir(parents=True, exist_ok=True)
    inference_model_path = directory / f'model_{generation}.jit.pt'
    traced = torch.jit.trace(_UniformChessModel(), torch.zeros((1, 29, 8, 8)))
    traced.save(str(inference_model_path))
    return CheckpointReference(
        generation=generation,
        manifest_path=checkpoint_manifest_path(generation, directory),
        model_path=directory / f'model_{generation}.pt',
        optimizer_path=directory / f'optimizer_{generation}.pt',
        inference_model_path=inference_model_path,
        inference_model_sha256=hashlib.sha256(inference_model_path.read_bytes()).hexdigest(),
    )


def _applied_policy(directory: Path, stopping: SearchStoppingConfiguration) -> SearchStopPolicy:
    generator = np.random.default_rng(20260901)
    features = generator.normal(size=(600, STOP_PREDICTOR_FEATURE_COUNT)).astype(np.float32)
    labels = (features[:, 0] > 0.0).astype(np.float32)
    groups = (np.arange(600) // 6).astype(np.uint64)
    fit = fit_stop_predictor(features, labels, groups)
    assert fit.network is not None
    predictor_path = directory / 'stop-predictor.jit.pt'
    predictor_sha256 = export_stop_predictor(fit.network, predictor_path)
    return SearchStopPolicy(
        checkpoint_multiples=tuple(stopping.checkpoint_multiples),
        thresholds=(0.5,) * len(stopping.checkpoint_multiples),
        movement_guard_epsilon=stopping.movement_guard_epsilon,
        cap_multiple=stopping.cap_multiple,
        predictor_path=predictor_path,
        predictor_sha256=predictor_sha256,
        apply_learned=True,
    )


def test_run_batch_survives_the_closed_open_closed_policy_flips(tmp_path: Path) -> None:
    configuration = _configuration()
    game = ChessImplementation(configuration)
    stopping = configuration.training.lifecycle.search_stopping
    inbox = tmp_path / 'completed-games' / 'inbox'
    inbox.mkdir(parents=True)
    worker = SelfPlayWorker(
        game=game,
        parallel_game_count=_GAME_COUNT,
        worker_id=0,
        device_id=0,
        inbox_path=inbox,
    )
    try:
        # Warmup generation: closed policy, audits still search to the cap.
        worker.refresh_published_model(_checkpoint(tmp_path, 0), closed_policy(stopping))
        worker.run_batch()
        assert sum(len(active.observations) for active in worker.active_games) >= 1

        # The production flip that crash-looped v21: the first applied publication.
        worker.refresh_published_model(_checkpoint(tmp_path, 1), _applied_policy(tmp_path, stopping))
        worker.run_batch()
        assert sum(len(active.observations) for active in worker.active_games) >= 1
        second_batch = [active.observations[-1] for active in worker.active_games if active.observations]
        assert all(
            observation.final_visits - observation.starting_visits
            <= int(stopping.cap_multiple * _BASELINE_VISITS + 0.5)
            for observation in second_batch
        )

        # Fail-closed flip back: requests must drop their checkpoints again.
        worker.refresh_published_model(_checkpoint(tmp_path, 2), closed_policy(stopping))
        worker.run_batch()

        # The audit stream materialized raw records for at least one generation.
        stopping_path = tmp_path / 'search-stopping'
        audit_files = sorted(stopping_path.glob('audit-generation-*-worker-*.np'))
        assert audit_files, 'audit searches produced no records'
        dtype = audit_record_dtype(len(stopping.checkpoint_multiples))
        records = np.concatenate([read_records(path, dtype) for path in audit_files])
        assert records.shape[0] >= 1
        assert np.isfinite(records['kl_to_final']).all()
        assert np.isfinite(records['features']).all()
    finally:
        worker.close()
