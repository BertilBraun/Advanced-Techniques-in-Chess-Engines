from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
import src.experiment.run as experiment_run
from src.evaluation.manager import EvaluationManager
from src.experiment.configuration import (
    ExperimentConfiguration,
    experiment_configuration_sha256,
    load_experiment_configuration,
)
from src.experiment.run import ExperimentRunManifest
from src.experiment.run_contract import ApprovalRecord, ResolvedHardware
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.representation import PackedPlaneLayout
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.retention import CheckpointRetention
from src.training.configuration import CreditTrainingParams
from src.training.credit_ledger import CreditLedger, CreditLedgerState
from src.training.targets import (
    NextPolicyHeadLayout,
    RemainingGameLengthHeadLayout,
    TrainingTargetLayout,
)
from src.util.atomic_file import write_text_atomically
from test_helpers.checkpoints import checkpoint_reference, materialized_checkpoint
from test_helpers.configuration_paths import REPOSITORY_CONFIG_DIRECTORY, TEST_CONFIG_DIRECTORY

PRODUCTION_CONFIGURATION_PATH = REPOSITORY_CONFIG_DIRECTORY / 'production' / 'vast-chess-8gpu-optimal.yaml'
ELEVEN_HOURS_SECONDS = 11.0 * 3600.0
SIXTEEN_HOURS_SECONDS = 16.0 * 3600.0


def _production_configuration() -> ChessExperimentConfiguration:
    loaded = load_experiment_configuration(PRODUCTION_CONFIGURATION_PATH)
    assert isinstance(loaded, ChessExperimentConfiguration)
    return loaded


def _credit_parameters() -> CreditTrainingParams:
    return _production_configuration().training.lifecycle.credit


def _replay_layout() -> ReplayLayout:
    return ReplayLayout(
        packed_planes=PackedPlaneLayout(board_size=8, binary_plane_count=2, scalar_count=1),
        targets=TrainingTargetLayout(
            action_size=100,
            wdl_size=3,
            auxiliary_heads=(
                NextPolicyHeadLayout(kind='next_policy', action_size=100, ply_offset=1),
                RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=400.0),
            ),
        ),
        maximum_policy_entries=60,
        maximum_legal_actions=100,
    )


def test_credit_ledger_resumes_its_generation_and_ignores_the_gen_zero_starting_checkpoint(tmp_path: Path) -> None:
    parameters = _credit_parameters()
    materialized_checkpoint(tmp_path, 0)
    resumed_checkpoint = materialized_checkpoint(tmp_path, 230)
    state = CreditLedgerState(
        completed_optimizer_steps=230 * parameters.optimizer_steps_per_quantum,
        earned_credits=Decimal(10_000_000),
        consumed_credits=Decimal(9_000_000),
        active_checkpoint=resumed_checkpoint,
    )
    write_text_atomically(tmp_path / 'credit-ledger.json', state.model_dump_json(indent=2) + '\n')

    ledger = CreditLedger(
        tmp_path,
        parameters,
        2048,
        materialized_checkpoint(tmp_path, 0),
        adopt_completed_quantum=False,
    )

    assert ledger.model_generation == 230
    assert ledger.state.active_checkpoint.generation == 230
    assert ledger.state.available_credits == Decimal(1_000_000)


def _evaluation_experiment(run_path: Path) -> ChessExperimentConfiguration:
    loaded = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    assert isinstance(loaded, ChessExperimentConfiguration)
    return loaded.model_copy(update={'training': loaded.training.model_copy(update={'save_path': str(run_path)})})


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_elapsed_run_time_survives_a_restart(tmp_path: Path) -> None:
    experiment = _evaluation_experiment(tmp_path)
    first_clock = _FakeClock()
    first = EvaluationManager(experiment, checkpoint_reference(tmp_path, 0), first_clock)
    first_clock.now = ELEVEN_HOURS_SECONDS
    first.close()

    second_clock = _FakeClock()
    second_clock.now = 5_000_000.0
    second = EvaluationManager(experiment, checkpoint_reference(tmp_path, 0), second_clock)

    assert second.elapsed_seconds == pytest.approx(ELEVEN_HOURS_SECONDS)
    second_clock.now += SIXTEEN_HOURS_SECONDS - ELEVEN_HOURS_SECONDS
    assert second.elapsed_seconds == pytest.approx(SIXTEEN_HOURS_SECONDS)


def test_replay_store_reopens_after_a_configuration_change(tmp_path: Path) -> None:
    layout = _replay_layout()
    path = tmp_path / 'replay.bin'
    created = ReplayStore.create(path, layout, 4_096, 2_048)
    created.close()

    reopened = ReplayStore.open(path, layout)
    try:
        assert reopened.state.maximum_capacity == 4_096
    finally:
        reopened.close()


def test_generation_zero_checkpoint_survives_production_retention_and_still_loads(tmp_path: Path) -> None:
    generations = (*range(0, 12), *range(224, 236))
    for generation in generations:
        materialized_checkpoint(tmp_path, generation)

    CheckpointRetention(tmp_path, _production_configuration().training.lifecycle).apply(
        active_generation=235,
        required_inference_generations=(232, 233, 234),
    )

    assert CheckpointReference.load(tmp_path, 0).generation == 0


def _run_manifest(experiment: ExperimentConfiguration) -> ExperimentRunManifest:
    return ExperimentRunManifest(
        experiment=experiment,
        approval=ApprovalRecord(
            approved_by='owner',
            approved_at_utc=datetime(2026, 8, 25, tzinfo=timezone.utc),
            source_revision='0' * 40,
            configuration_sha256=experiment_configuration_sha256(experiment),
            maximum_cost=None,
        ),
        resolved_hardware=ResolvedHardware(
            visible_gpu_names=('NVIDIA GeForce RTX 4070 SUPER',) * 8,
            visible_gpu_count=8,
            logical_cpu_count=80,
            total_ram_gib=200.0,
            free_disk_gib=100.0,
        ),
        source_revision='0' * 40,
        source_worktree_clean=True,
        initial_generation=0,
        initial_model_sha256='a' * 64,
        evaluation_dataset_sha256='b' * 64,
        evaluation_dataset_manifest_sha256='c' * 64,
        opening_suite_manifest_sha256='d' * 64,
        evaluation_engine_artifact_sha256=('e' * 64,),
        open_file_soft_limit=65536,
        torch_version='2.12.1+cu126',
        cuda_version='12.6',
    )


def test_run_manifest_archives_the_previous_configuration(tmp_path: Path) -> None:
    original = _production_configuration()
    manifest_path = tmp_path / 'run_manifest.json'
    experiment_run._write_manifest(manifest_path, _run_manifest(original))
    changed = original.model_copy(
        update={'run': original.run.model_copy(update={'run_name': 'vast-chess-8gpu-optimal-restarted'})}
    )

    changed_manifest = _run_manifest(changed)
    written = experiment_run._write_manifest(manifest_path, changed_manifest)

    assert written == changed_manifest
    assert len(tuple((tmp_path / 'run_manifests').glob('run_manifest-*.json'))) == 1
    reloaded = ExperimentRunManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    assert reloaded.experiment.run.run_name == 'vast-chess-8gpu-optimal-restarted'
