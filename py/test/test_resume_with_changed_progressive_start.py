"""Probe: can a live run be stopped, its progressive rung threshold lowered, and resumed in place?"""

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
from src.replay.description import ReplayDescription
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.retention import CheckpointRetention
from src.training.configuration import CreditTrainingParams
from src.training.credit_ledger import CreditLedger, CreditLedgerState
from src.training.progressive import (
    ProgressiveModelSizingConfiguration,
    ProgressiveTrainingStateStore,
)
from src.training.targets import (
    NextPolicyHeadLayout,
    RemainingGameLengthHeadLayout,
    TrainingTargetLayout,
)
from src.util.atomic_file import write_text_atomically
from test_helpers.checkpoints import materialized_checkpoint
from test_helpers.configuration_paths import REPOSITORY_CONFIG_DIRECTORY, TEST_CONFIG_DIRECTORY

PRODUCTION_CONFIGURATION_PATH = REPOSITORY_CONFIG_DIRECTORY / 'production' / 'vast-chess-8gpu-optimal.yaml'
ELEVEN_HOURS_SECONDS = 11.0 * 3600.0
SIXTEEN_HOURS_SECONDS = 16.0 * 3600.0
LOWERED_START_DAYS = Decimal('0.667')
BASELINE_START_DAYS = Decimal('1.0')


def _production_configuration() -> ChessExperimentConfiguration:
    # Pinned to the pre-change threshold rather than whatever the shipped file currently says, so the
    # probe keeps comparing two different configurations after the production value moves.
    loaded = load_experiment_configuration(PRODUCTION_CONFIGURATION_PATH)
    assert isinstance(loaded, ChessExperimentConfiguration)
    baseline = _with_second_rung_start(loaded, BASELINE_START_DAYS)
    assert isinstance(baseline, ChessExperimentConfiguration)
    return baseline


def _with_second_rung_start(configuration: ExperimentConfiguration, days: Decimal) -> ExperimentConfiguration:
    sizing = configuration.training.progressive_model_sizing
    first, second, *rest = sizing.models
    lowered = sizing.model_copy(
        update={'models': (first, second.model_copy(update={'training_start_days': days}), *rest)}
    )
    return configuration.model_copy(
        update={'training': configuration.training.model_copy(update={'progressive_model_sizing': lowered})}
    )


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


def test_lowering_the_second_rung_changes_the_approval_configuration_sha256() -> None:
    original = _production_configuration()
    lowered = _with_second_rung_start(original, LOWERED_START_DAYS)

    assert experiment_configuration_sha256(original) != experiment_configuration_sha256(lowered)


def test_second_rung_is_ineligible_at_eleven_hours_and_eligible_at_sixteen() -> None:
    sizing = _with_second_rung_start(_production_configuration(), LOWERED_START_DAYS).training.progressive_model_sizing

    assert sizing.eligible_model_ids(ELEVEN_HOURS_SECONDS) == ('chess-attention-1m',)
    assert sizing.eligible_model_ids(SIXTEEN_HOURS_SECONDS + 60.0) == (
        'chess-attention-1m',
        'chess-attention-2m',
    )


def test_original_second_rung_is_ineligible_at_sixteen_hours() -> None:
    sizing = _production_configuration().training.progressive_model_sizing

    assert sizing.eligible_model_ids(SIXTEEN_HOURS_SECONDS + 60.0) == ('chess-attention-1m',)


def _progressive_store(path: Path, sizing: ProgressiveModelSizingConfiguration) -> ProgressiveTrainingStateStore:
    return ProgressiveTrainingStateStore(path, sizing)


def test_progressive_state_reopens_under_the_lowered_threshold(tmp_path: Path) -> None:
    original = _production_configuration()
    state_path = tmp_path / 'progressive-training.json'
    before = _progressive_store(state_path, original.training.progressive_model_sizing)
    persisted = state_path.read_text(encoding='utf-8')

    after = _progressive_store(
        state_path, _with_second_rung_start(original, LOWERED_START_DAYS).training.progressive_model_sizing
    )

    assert state_path.read_text(encoding='utf-8') == persisted
    assert after.state == before.state
    assert after.state.active_model_id == 'chess-attention-1m'


def test_restarted_quantum_admits_the_second_rung_once_elapsed_passes_the_lowered_threshold(tmp_path: Path) -> None:
    original = _production_configuration()
    state_path = tmp_path / 'progressive-training.json'
    replay = ReplayDescription(
        path=tmp_path / 'replay.bin',
        head=0,
        size=1_000,
        logical_capacity=2_000,
        maximum_capacity=2_000,
        layout=_replay_layout(),
    )

    before = _progressive_store(state_path, original.training.progressive_model_sizing)
    pending_before = before.begin_quantum(ELEVEN_HOURS_SECONDS, replay, 115_000, 500)
    assert pending_before.required_model_ids == ('chess-attention-1m',)

    # A checkpoint-safe stop happens between quanta, so the persisted state carries no pending quantum.
    write_text_atomically(
        state_path,
        before.state.model_copy(update={'pending_quantum': None}).model_dump_json(indent=2) + '\n',
    )

    after = _progressive_store(
        state_path, _with_second_rung_start(original, LOWERED_START_DAYS).training.progressive_model_sizing
    )
    pending_after = after.begin_quantum(SIXTEEN_HOURS_SECONDS + 60.0, replay, 115_500, 500)

    assert pending_after.required_model_ids == ('chess-attention-1m', 'chess-attention-2m')


def test_pending_quantum_from_an_unclean_stop_pins_the_old_rung_set(tmp_path: Path) -> None:
    original = _production_configuration()
    state_path = tmp_path / 'progressive-training.json'
    replay = ReplayDescription(
        path=tmp_path / 'replay.bin',
        head=0,
        size=1_000,
        logical_capacity=2_000,
        maximum_capacity=2_000,
        layout=_replay_layout(),
    )
    before = _progressive_store(state_path, original.training.progressive_model_sizing)
    before.begin_quantum(ELEVEN_HOURS_SECONDS, replay, 115_000, 500)

    after = _progressive_store(
        state_path, _with_second_rung_start(original, LOWERED_START_DAYS).training.progressive_model_sizing
    )
    pending_after = after.begin_quantum(SIXTEEN_HOURS_SECONDS + 60.0, replay, 115_000, 500)

    assert pending_after.required_model_ids == ('chess-attention-1m',)


def test_pending_quantum_rejects_a_changed_replay_batch_across_restart(tmp_path: Path) -> None:
    original = _production_configuration()
    state_path = tmp_path / 'progressive-training.json'
    layout = _replay_layout()
    replay = ReplayDescription(
        path=tmp_path / 'replay.bin',
        head=0,
        size=1_000,
        logical_capacity=2_000,
        maximum_capacity=2_000,
        layout=layout,
    )
    before = _progressive_store(state_path, original.training.progressive_model_sizing)
    before.begin_quantum(ELEVEN_HOURS_SECONDS, replay, 115_000, 500)

    after = _progressive_store(
        state_path, _with_second_rung_start(original, LOWERED_START_DAYS).training.progressive_model_sizing
    )
    grown = replay.model_copy(update={'size': 1_100})

    with pytest.raises(ValueError, match='replay batches changed across restart'):
        after.begin_quantum(SIXTEEN_HOURS_SECONDS, grown, 115_000, 500)


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
    from test_helpers.checkpoints import checkpoint_reference

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


def test_replay_store_reopens_after_the_configuration_change(tmp_path: Path) -> None:
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


def test_run_manifest_accepts_the_changed_configuration_and_archives_the_previous_one(tmp_path: Path) -> None:
    original = _production_configuration()
    manifest_path = tmp_path / 'run_manifest.json'
    experiment_run._write_manifest(manifest_path, _run_manifest(original))

    lowered_manifest = _run_manifest(_with_second_rung_start(original, LOWERED_START_DAYS))
    written = experiment_run._write_manifest(manifest_path, lowered_manifest)

    assert written == lowered_manifest
    assert len(tuple((tmp_path / 'run_manifests').glob('run_manifest-*.json'))) == 1
    reloaded = ExperimentRunManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    assert reloaded.experiment.training.progressive_model_sizing.models[1].training_start_days == LOWERED_START_DAYS
