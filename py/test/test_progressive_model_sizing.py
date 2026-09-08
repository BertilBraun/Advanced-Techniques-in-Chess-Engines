from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from src.evaluation.ladder import PrimaryLadderEloObservation
from src.experiment.configuration import ExperimentConfiguration, load_experiment_configuration
from src.games.implementation import GameImplementation
from src.games.representation import NetworkDimensions, PackedPlaneLayout
from src.replay.description import ReplayDescription
from src.replay.layout import ReplayLayout
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.contracts import CheckpointManifest
from src.training.checkpoint.persistence import publish_checkpoint
from src.training.configuration import TrainingArgs
from src.training.network import (
    AttentionNetworkParams,
    DisabledResidualContext,
    GoPointPassPolicyHeadConfiguration,
    NetworkDefinition,
    NetworkParams,
)
from src.training.progressive import (
    CompletedCandidateTraining,
    ElapsedCandidateStartConfiguration,
    EloPlateauCandidateStartConfiguration,
    FixedModelSizingConfiguration,
    ProgressiveModelDefinition,
    ProgressiveModelSizingConfiguration,
    ProgressiveTrainingStateStore,
    TotalLossEmaPromotionConfiguration,
    retain_progressive_candidate_checkpoints,
)
from src.training.session import ProgressiveTrainingSession
from src.training.targets import TrainingTargetLayout
from src.training.trainer import TrainerGroup
from src.training.trainer.contracts import TrainerStartup
from src.util.atomic_file import write_text_atomically
from test_helpers.checkpoints import checkpoint_reference
from test_helpers.configuration_paths import TEST_CONFIG_DIRECTORY


def _network(width: int) -> NetworkParams:
    return NetworkParams(
        num_layers=2,
        hidden_size=width,
        residual_context=DisabledResidualContext(),
        policy_head=GoPointPassPolicyHeadConfiguration(),
        num_value_channels=1,
        value_fc_size=8,
    )


def _configuration(warmup_quanta: int = 2) -> ProgressiveModelSizingConfiguration:
    return ProgressiveModelSizingConfiguration(
        kind='progressive',
        models=(
            ProgressiveModelDefinition(model_id='small', network=_network(8)),
            ProgressiveModelDefinition(model_id='medium', network=_network(12)),
            ProgressiveModelDefinition(model_id='large', network=_network(16)),
        ),
        candidate_start=EloPlateauCandidateStartConfiguration(
            kind='elo_plateau',
            minimum_worthwhile_gain_per_hour=5.0,
        ),
        promotion=TotalLossEmaPromotionConfiguration(
            decay=0.5,
            warmup_quanta=warmup_quanta,
            maximum_relative_loss=1.01,
            candidate_catchup_learning_rate=0.005,
        ),
    )


def _elapsed_configuration() -> ProgressiveModelSizingConfiguration:
    return _configuration().validated_copy(
        update={
            'candidate_start': ElapsedCandidateStartConfiguration(
                kind='elapsed',
                start_days=(0.75, 1.5),
            ).model_dump(mode='json')
        }
    )


def test_progressive_model_definition_accepts_attention_architecture() -> None:
    definition = ProgressiveModelDefinition(
        model_id='attention',
        network=AttentionNetworkParams(
            num_layers=2,
            embedding_size=64,
            num_heads=4,
            feedforward_size=128,
            policy_head=GoPointPassPolicyHeadConfiguration(),
        ),
    )

    assert isinstance(definition.network, AttentionNetworkParams)


def _replay(tmp_path: Path, head: int = 3) -> ReplayDescription:
    return ReplayDescription(
        path=tmp_path / 'replay.bin',
        head=head,
        size=32,
        logical_capacity=64,
        maximum_capacity=128,
        layout=ReplayLayout(
            packed_planes=PackedPlaneLayout(board_size=7, binary_plane_count=2, scalar_count=1),
            targets=TrainingTargetLayout(action_size=50, wdl_size=3, auxiliary_heads=()),
            maximum_policy_entries=50,
            maximum_legal_actions=50,
        ),
    )


def _checkpoint(tmp_path: Path, model_id: str, generation: int) -> CheckpointReference:
    return checkpoint_reference(tmp_path / 'models' / model_id, generation)


def _latch_candidate_start(store: ProgressiveTrainingStateStore) -> None:
    store.observe_primary_ladder_elos((PrimaryLadderEloObservation(boundary_seconds=3600, elo=4.9),))


@pytest.mark.parametrize('model_count', (2, 4))
def test_model_schedule_accepts_any_nonempty_model_tuple(model_count: int) -> None:
    models = tuple(
        ProgressiveModelDefinition(
            model_id=f'model-{index}',
            network=_network(8 + index),
        )
        for index in range(model_count)
    )

    configuration = ProgressiveModelSizingConfiguration(
        kind='progressive',
        models=models,
        candidate_start=EloPlateauCandidateStartConfiguration(
            kind='elo_plateau',
            minimum_worthwhile_gain_per_hour=5.0,
        ),
        promotion=TotalLossEmaPromotionConfiguration(
            decay=0.9,
            warmup_quanta=2,
            maximum_relative_loss=1.01,
            candidate_catchup_learning_rate=0.004,
        ),
    )

    assert configuration.models == models
    assert configuration.is_progressive


def test_fixed_model_configuration_has_no_candidate_or_promotion_policy() -> None:
    configuration = FixedModelSizingConfiguration(
        kind='fixed',
        model=ProgressiveModelDefinition(model_id='only', network=_network(8)),
    )

    assert configuration.models == (configuration.model,)
    assert not configuration.is_progressive
    assert 'candidate_start' not in type(configuration).model_fields
    assert 'promotion' not in type(configuration).model_fields


@pytest.mark.parametrize(
    ('elapsed_seconds', 'expected'),
    (
        (0.0, ('small',)),
        (64_799.9, ('small',)),
        (64_800.0, ('small', 'medium')),
        (129_600.0, ('small', 'medium', 'large')),
    ),
)
def test_elapsed_start_policy_preserves_timed_candidate_eligibility(
    tmp_path: Path,
    elapsed_seconds: float,
    expected: tuple[str, ...],
) -> None:
    store = ProgressiveTrainingStateStore(tmp_path / 'state.json', _elapsed_configuration())

    assert store.begin_quantum(elapsed_seconds, _replay(tmp_path), 0, 4).required_model_ids == expected


def test_elapsed_start_policy_requires_one_start_per_candidate() -> None:
    with pytest.raises(ValueError, match='one entry per candidate'):
        _configuration().validated_copy(
            update={
                'candidate_start': ElapsedCandidateStartConfiguration(
                    kind='elapsed',
                    start_days=(0.75,),
                ).model_dump(mode='json')
            }
        )


def test_model_schedule_rejects_an_empty_model_tuple() -> None:
    with pytest.raises(ValueError):
        ProgressiveModelSizingConfiguration(
            kind='progressive',
            models=(),
            candidate_start=EloPlateauCandidateStartConfiguration(
                kind='elo_plateau',
                minimum_worthwhile_gain_per_hour=5.0,
            ),
            promotion=TotalLossEmaPromotionConfiguration(
                decay=0.9,
                warmup_quanta=2,
                maximum_relative_loss=1.01,
                candidate_catchup_learning_rate=0.004,
            ),
        )


def test_training_configuration_has_one_network_owner() -> None:
    assert 'network' not in TrainingArgs.model_fields
    assert TrainingArgs.model_fields['progressive_model_sizing'].is_required()


def test_candidate_start_ema_has_a_persisted_zero_baseline(tmp_path: Path) -> None:
    state_path = tmp_path / 'progressive-training.json'
    store = ProgressiveTrainingStateStore(state_path, _configuration())

    assert store.state.candidate_start.latest_boundary_seconds == 0
    assert store.state.candidate_start.ema_observations == 0
    assert store.state.candidate_start.ema_elo == 0.0
    assert store.state.candidate_start.instantaneous_ema_gain_per_hour is None
    assert not store.state.candidate_start.latched
    assert json.loads(state_path.read_text(encoding='utf-8'))['schema_version'] == 3


def test_candidate_starts_only_below_the_gain_threshold(tmp_path: Path) -> None:
    exact = ProgressiveTrainingStateStore(tmp_path / 'exact.json', _configuration())
    exact_updates = exact.observe_primary_ladder_elos((PrimaryLadderEloObservation(boundary_seconds=3600, elo=5.0),))

    assert exact_updates[0].instantaneous_ema_gain_per_hour == pytest.approx(5.0)
    assert not exact.state.candidate_start.latched
    assert exact.begin_quantum(100_000.0, _replay(tmp_path), 0, 4).required_model_ids == ('small',)

    below = ProgressiveTrainingStateStore(tmp_path / 'below.json', _configuration())
    below.observe_primary_ladder_elos((PrimaryLadderEloObservation(boundary_seconds=3600, elo=4.9),))

    assert below.state.candidate_start.latched
    assert below.begin_quantum(0.0, _replay(tmp_path), 0, 4).required_model_ids == ('small', 'medium')


def test_candidate_start_ema_matches_tensorboard_bias_correction(tmp_path: Path) -> None:
    store = ProgressiveTrainingStateStore(tmp_path / 'state.json', _configuration())

    updates = store.observe_primary_ladder_elos(
        (
            PrimaryLadderEloObservation(boundary_seconds=1800, elo=800.0),
            PrimaryLadderEloObservation(boundary_seconds=3600, elo=1000.0),
            PrimaryLadderEloObservation(boundary_seconds=5400, elo=1200.0),
        )
    )

    raw_ema = 0.05 * 1200.0 + 0.95 * (0.05 * 1000.0 + 0.95 * 0.05 * 800.0)
    expected_ema = raw_ema / (1.0 - 0.95**3)
    previous_raw_ema = 0.05 * 1000.0 + 0.95 * 0.05 * 800.0
    previous_ema = previous_raw_ema / (1.0 - 0.95**2)

    assert updates[-1].ema_observations == 3
    assert updates[-1].ema_elo == pytest.approx(expected_ema)
    assert updates[-1].instantaneous_ema_gain_per_hour == pytest.approx((expected_ema - previous_ema) / 0.5)


def test_candidate_start_latch_never_clears(tmp_path: Path) -> None:
    store = ProgressiveTrainingStateStore(tmp_path / 'state.json', _configuration())
    store.observe_primary_ladder_elos((PrimaryLadderEloObservation(boundary_seconds=3600, elo=4.9),))
    store.observe_primary_ladder_elos((PrimaryLadderEloObservation(boundary_seconds=7200, elo=5000.0),))

    assert store.state.candidate_start.instantaneous_ema_gain_per_hour is not None
    assert store.state.candidate_start.instantaneous_ema_gain_per_hour > 5.0
    assert store.state.candidate_start.latched


def test_candidate_start_recovers_and_ignores_duplicate_or_older_boundaries(tmp_path: Path) -> None:
    state_path = tmp_path / 'state.json'
    first = ProgressiveTrainingStateStore(state_path, _configuration())
    first.observe_primary_ladder_elos((PrimaryLadderEloObservation(boundary_seconds=3600, elo=100.0),))

    restarted = ProgressiveTrainingStateStore(state_path, _configuration())
    updates = restarted.observe_primary_ladder_elos(
        (
            PrimaryLadderEloObservation(boundary_seconds=7200, elo=190.0),
            PrimaryLadderEloObservation(boundary_seconds=1800, elo=5000.0),
            PrimaryLadderEloObservation(boundary_seconds=3600, elo=100.0),
        )
    )

    assert tuple(update.latest_boundary_seconds for update in updates) == (7200,)
    assert restarted.state.candidate_start.ema_observations == 2
    assert restarted.state.candidate_start.ema_elo == pytest.approx(146.15384615384616)
    assert restarted.state.candidate_start.latest_boundary_seconds == 7200
    assert not restarted.state.candidate_start.latched


def test_pending_quantum_persists_replay_identity_and_candidate_completion(tmp_path: Path) -> None:
    state_path = tmp_path / 'progressive-training.json'
    store = ProgressiveTrainingStateStore(state_path, _configuration())
    _latch_candidate_start(store)
    store.initialize_candidate('small', 40, _checkpoint(tmp_path, 'small', 10))
    pending = store.begin_quantum(0.0, _replay(tmp_path), 40, 4)

    assert pending.required_model_ids == ('small', 'medium')
    assert pending.replay_batch.replay == _replay(tmp_path)
    assert pending.replay_batch.source_optimizer_steps == 40
    persisted_replay_batch = json.loads(state_path.read_text(encoding='utf-8'))['pending_quantum']['replay_batch']
    assert set(persisted_replay_batch) == {'replay', 'source_optimizer_steps'}
    store.record_candidate(
        CompletedCandidateTraining(
            model_id='small',
            completed_optimizer_steps=44,
            checkpoint=_checkpoint(tmp_path, 'small', 11),
            comparable_total_loss=2.0,
        )
    )

    restarted = ProgressiveTrainingStateStore(state_path, _configuration())
    resumed = restarted.begin_quantum(100_000.0, _replay(tmp_path), 40, 4)

    assert resumed.next_model_id == 'medium'
    assert resumed.completed[0].comparable_total_loss == 2.0
    with pytest.raises(ValueError, match='replay batches changed'):
        restarted.begin_quantum(100_000.0, _replay(tmp_path, head=4), 40, 4)


def test_candidate_latch_takes_effect_at_the_next_quantum(tmp_path: Path) -> None:
    store = ProgressiveTrainingStateStore(tmp_path / 'state.json', _configuration())
    replay = _replay(tmp_path)
    current = store.begin_quantum(0.0, replay, 0, 4)

    _latch_candidate_start(store)

    assert current.required_model_ids == ('small',)
    assert store.state.pending_quantum is not None
    assert store.state.pending_quantum.required_model_ids == ('small',)
    store.record_candidate(
        CompletedCandidateTraining(
            model_id='small',
            completed_optimizer_steps=4,
            checkpoint=_checkpoint(tmp_path, 'small', 1),
            comparable_total_loss=1.0,
        )
    )
    store.complete_quantum()

    assert store.begin_quantum(0.0, replay, 4, 4).required_model_ids == ('small', 'medium')


def test_promotion_requires_warmup_and_one_percent_comparable_ema(tmp_path: Path) -> None:
    store = ProgressiveTrainingStateStore(tmp_path / 'state.json', _configuration(warmup_quanta=2))
    _latch_candidate_start(store)
    replay = _replay(tmp_path)

    for quantum, (active_loss, candidate_loss) in enumerate(((2.0, 2.01), (1.8, 1.815))):
        source_steps = quantum * 4
        store.begin_quantum(0.0, replay, source_steps, 4)
        store.record_candidate(
            CompletedCandidateTraining(
                model_id='small',
                completed_optimizer_steps=source_steps + 4,
                checkpoint=_checkpoint(tmp_path, 'small', quantum + 1),
                comparable_total_loss=active_loss,
            )
        )
        store.record_candidate(
            CompletedCandidateTraining(
                model_id='medium',
                completed_optimizer_steps=source_steps + 4,
                checkpoint=_checkpoint(tmp_path, 'medium', quantum + 1),
                comparable_total_loss=candidate_loss,
            )
        )
        active_model_id = store.complete_quantum()

    assert active_model_id == 'medium'


def test_later_candidate_is_not_skipped_after_first_promotion(tmp_path: Path) -> None:
    store = ProgressiveTrainingStateStore(tmp_path / 'state.json', _configuration(warmup_quanta=1))
    _latch_candidate_start(store)
    replay = _replay(tmp_path)
    store.begin_quantum(0.0, replay, 0, 4)
    for model_id in ('small', 'medium'):
        store.record_candidate(
            CompletedCandidateTraining(
                model_id=model_id,
                completed_optimizer_steps=4,
                checkpoint=_checkpoint(tmp_path, model_id, 1),
                comparable_total_loss=1.0,
            )
        )
    assert store.complete_quantum() == 'medium'

    pending = store.begin_quantum(0.0, replay, 4, 4)

    assert pending.required_model_ids == ('medium', 'large')


def test_publication_relabels_private_candidate_generation_atomically(tmp_path: Path) -> None:
    private_path = tmp_path / 'models' / 'medium'
    private_path.mkdir(parents=True)
    source = _checkpoint(tmp_path, 'medium', 2)
    for path, payload in (
        (source.model_path, b'model'),
        (source.optimizer_path, b'optimizer'),
        (source.inference_model_path, b'inference'),
    ):
        path.write_bytes(payload)
    manifest = CheckpointManifest(
        generation=2,
        network=NetworkDefinition(
            architecture=_network(12),
            dimensions=NetworkDimensions(channels=3, rows=7, columns=7, actions=50, outcomes=3),
            auxiliary_heads=(),
        ),
        model_path=source.model_path.name,
        model_sha256=hashlib.sha256(b'model').hexdigest(),
        optimizer_path=source.optimizer_path.name,
        optimizer_sha256=hashlib.sha256(b'optimizer').hexdigest(),
        inference_model_path=source.inference_model_path.name,
        inference_model_sha256=hashlib.sha256(b'inference').hexdigest(),
    )
    write_text_atomically(source.manifest_path, manifest.model_dump_json(indent=2) + '\n')

    published = publish_checkpoint(source, generation=7, destination_folder=tmp_path)

    assert published.generation == 7
    assert published.model_path.read_bytes() == b'model'
    assert publish_checkpoint(source, generation=7, destination_folder=tmp_path) == published


def test_candidate_retention_keeps_only_exact_restart_state(tmp_path: Path) -> None:
    store = ProgressiveTrainingStateStore(tmp_path / 'state.json', _configuration())
    retained = _checkpoint(tmp_path, 'small', 2)
    obsolete = _checkpoint(tmp_path, 'small', 1)
    for checkpoint in (retained, obsolete):
        checkpoint.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        for path in (
            checkpoint.manifest_path,
            checkpoint.model_path,
            checkpoint.optimizer_path,
            checkpoint.inference_model_path,
        ):
            path.write_bytes(b'x')
    store.initialize_candidate('small', 8, retained)

    retain_progressive_candidate_checkpoints(tmp_path, store.state)

    assert all(
        path.exists()
        for path in (
            retained.manifest_path,
            retained.model_path,
            retained.optimizer_path,
            retained.inference_model_path,
        )
    )
    assert not obsolete.manifest_path.exists()
    assert not obsolete.model_path.exists()
    assert not obsolete.optimizer_path.exists()
    assert not obsolete.inference_model_path.exists()


@dataclass
class _RecordingTrainerGroup:
    experiment: ExperimentConfiguration
    game: GameImplementation
    startup: TrainerStartup
    closed: bool = False

    def close(self) -> None:
        self.closed = True


def test_progressive_trainer_groups_remain_alive_across_quanta(tmp_path: Path) -> None:
    loaded = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    configuration = loaded.model_copy(
        update={
            'training': loaded.training.model_copy(
                update={
                    'save_path': str(tmp_path),
                    'progressive_model_sizing': _configuration(),
                }
            )
        }
    )
    network = configuration.training.progressive_model_sizing.models[0].network
    created: list[_RecordingTrainerGroup] = []

    def trainer_group_factory(
        experiment: ExperimentConfiguration,
        game: GameImplementation,
        startup: TrainerStartup,
    ) -> TrainerGroup:
        trainer = _RecordingTrainerGroup(experiment, game, startup)
        created.append(trainer)
        return cast(TrainerGroup, trainer)

    session = ProgressiveTrainingSession(
        configuration,
        cast(GameImplementation, object()),
        trainer_group_factory,
    )

    first = session._trainer_group('small', network, tmp_path / 'small', 0)
    repeated = session._trainer_group('small', network, tmp_path / 'small', 1)
    second = session._trainer_group('large', network, tmp_path / 'large', 0)

    assert first is repeated
    assert first is not second
    assert len(created) == 2
    session.close()
    assert all(trainer.closed for trainer in created)
    assert session.trainers == {}


def test_progressive_learning_rate_uses_catchup_until_promotion(tmp_path: Path) -> None:
    loaded = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    configuration = loaded.model_copy(
        update={
            'training': loaded.training.model_copy(
                update={
                    'save_path': str(tmp_path),
                    'progressive_model_sizing': _configuration(),
                }
            )
        }
    )
    session = ProgressiveTrainingSession(configuration, cast(GameImplementation, object()))

    assert session._candidate_learning_rate('large', 500) == pytest.approx(0.005)
    session.state.state = session.state.state.validated_copy(update={'active_model_id': 'large'})
    assert session._candidate_learning_rate('large', 500) == pytest.approx(0.002)
