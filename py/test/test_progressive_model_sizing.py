from dataclasses import dataclass
from decimal import Decimal
import hashlib
from pathlib import Path

import pytest

from src.games.representation import NetworkDimensions, PackedPlaneLayout
from src.replay.layout import ReplayLayout
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.contracts import CheckpointManifest
from src.training.checkpoint.persistence import publish_checkpoint
from src.training.network import AttentionNetworkParams, DisabledResidualContext, NetworkDefinition, NetworkParams
from src.training.progressive import (
    CompletedCandidateTraining,
    ProgressiveModelDefinition,
    ProgressiveModelSizingConfiguration,
    ProgressiveTrainingStateStore,
    TotalLossEmaPromotionConfiguration,
    retain_progressive_candidate_checkpoints,
)
from src.util.atomic_file import write_text_atomically
from src.training.targets import TrainingTargetLayout


def _network(width: int) -> NetworkParams:
    return NetworkParams(
        num_layers=2,
        hidden_size=width,
        residual_context=DisabledResidualContext(),
        num_policy_channels=2,
        num_value_channels=1,
        value_fc_size=8,
    )


def _configuration(warmup_quanta: int = 2) -> ProgressiveModelSizingConfiguration:
    return ProgressiveModelSizingConfiguration(
        models=(
            ProgressiveModelDefinition(model_id='small', training_start_days=Decimal('0.0'), network=_network(8)),
            ProgressiveModelDefinition(model_id='medium', training_start_days=Decimal('0.75'), network=_network(12)),
            ProgressiveModelDefinition(model_id='large', training_start_days=Decimal('1.5'), network=_network(16)),
        ),
        promotion=TotalLossEmaPromotionConfiguration(
            decay=0.5,
            warmup_quanta=warmup_quanta,
            maximum_relative_loss=1.01,
        ),
    )


def test_progressive_model_definition_accepts_attention_architecture() -> None:
    definition = ProgressiveModelDefinition(
        model_id='attention',
        training_start_days=Decimal('0.0'),
        network=AttentionNetworkParams(
            num_layers=2,
            embedding_size=64,
            num_heads=4,
            feedforward_size=128,
        ),
    )

    assert isinstance(definition.network, AttentionNetworkParams)


@dataclass(frozen=True)
class ReplaySnapshot:
    path: Path
    head: int
    size: int
    logical_capacity: int
    maximum_capacity: int
    layout: ReplayLayout


def _replay(tmp_path: Path, head: int = 3) -> ReplaySnapshot:
    return ReplaySnapshot(
        path=tmp_path / 'replay.bin',
        head=head,
        size=32,
        logical_capacity=64,
        maximum_capacity=128,
        layout=ReplayLayout(
            packed_planes=PackedPlaneLayout(board_size=7, binary_plane_count=2, scalar_count=1),
            targets=TrainingTargetLayout(action_size=50, wdl_size=3, auxiliary_heads=()),
            maximum_policy_entries=50,
        ),
    )


def _checkpoint(tmp_path: Path, model_id: str, generation: int) -> CheckpointReference:
    root = tmp_path / 'models' / model_id
    return CheckpointReference(
        generation=generation,
        manifest_path=root / f'checkpoint_{generation}.json',
        model_path=root / f'model_{generation}.pt',
        optimizer_path=root / f'optimizer_{generation}.pt',
        inference_model_path=root / f'model_{generation}.jit.pt',
        inference_model_sha256=f'{generation:064x}',
    )


@pytest.mark.parametrize(
    ('elapsed_seconds', 'expected'),
    (
        (0.0, ('small',)),
        (64_799.9, ('small',)),
        (64_800.0, ('small', 'medium')),
        (129_600.0, ('small', 'medium', 'large')),
    ),
)
def test_elapsed_run_time_controls_candidate_eligibility(
    elapsed_seconds: float,
    expected: tuple[str, ...],
) -> None:
    assert _configuration().eligible_model_ids(elapsed_seconds) == expected


@pytest.mark.parametrize(
    'starts',
    (
        (Decimal('0.1'), Decimal('0.75'), Decimal('1.5')),
        (Decimal('0.0'), Decimal('0.75'), Decimal('0.75')),
    ),
)
def test_progressive_models_require_zero_then_strictly_increasing_starts(
    starts: tuple[Decimal, Decimal, Decimal],
) -> None:
    with pytest.raises(ValueError):
        ProgressiveModelSizingConfiguration(
            models=tuple(
                ProgressiveModelDefinition(model_id=f'model-{index}', training_start_days=start, network=_network(8))
                for index, start in enumerate(starts)
            ),
            promotion=TotalLossEmaPromotionConfiguration(decay=0.9, warmup_quanta=2),
        )


def test_pending_quantum_persists_replay_identity_and_candidate_completion(tmp_path: Path) -> None:
    state_path = tmp_path / 'progressive-training.json'
    store = ProgressiveTrainingStateStore(state_path, _configuration())
    store.initialize_candidate('small', 40, _checkpoint(tmp_path, 'small', 10))
    pending = store.begin_quantum(64_800.0, _replay(tmp_path), 40, 4)

    assert pending.required_model_ids == ('small', 'medium')
    store.record_candidate(
        CompletedCandidateTraining(
            model_id='small',
            completed_optimizer_steps=44,
            checkpoint=_checkpoint(tmp_path, 'small', 11),
            comparable_total_loss=2.0,
        )
    )

    restarted = ProgressiveTrainingStateStore(state_path, _configuration())
    resumed = restarted.begin_quantum(70_000.0, _replay(tmp_path), 40, 4)

    assert resumed.next_model_id == 'medium'
    assert resumed.completed[0].comparable_total_loss == 2.0
    with pytest.raises(ValueError, match='replay batches changed'):
        restarted.begin_quantum(70_000.0, _replay(tmp_path, head=4), 40, 4)


def test_promotion_requires_warmup_and_one_percent_comparable_ema(tmp_path: Path) -> None:
    store = ProgressiveTrainingStateStore(tmp_path / 'state.json', _configuration(warmup_quanta=2))
    replay = _replay(tmp_path)

    for quantum, (active_loss, candidate_loss) in enumerate(((2.0, 2.01), (1.8, 1.815))):
        source_steps = quantum * 4
        store.begin_quantum(64_800.0, replay, source_steps, 4)
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
    replay = _replay(tmp_path)
    store.begin_quantum(64_800.0, replay, 0, 4)
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

    pending = store.begin_quantum(129_600.0, replay, 4, 4)

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
            auxiliary_output_sizes=(),
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
