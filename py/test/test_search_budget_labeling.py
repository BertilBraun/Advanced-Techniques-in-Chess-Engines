from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from src.games.contracts import WdlTarget
from src.games.representation import PackedPlanePayload
from src.replay.contracts import (
    EligibleSearchBudgetTarget,
    IneligibleSearchBudgetTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.replay.label_source import ReplayLabelGameLocator
from src.replay.shard import ReplayShardGameMetadata, ReplayShardSourceGame
from src.search_budget.allocation import CurveAllocationIdentity, CurveAllocationPurpose
from src.search_budget.artifacts import load_persisted_model, write_persisted_model
from src.search_budget.calibration import initial_calibration_state
from src.search_budget.labeling import (
    DeepSearchRecord,
    DeepSearchShardArtifact,
    LabelGenerationSource,
    LabelPositionSource,
    LabelReplaySampleSource,
    PolicyCheckpointRecord,
    PredictionRecord,
    build_generation_source,
    candidate_allocations,
    checkpoint_visits_by_position,
    finalize_generation,
)
from src.search_budget.sampling import LabelPositionIdentity
from src.self_play.completed_game import (
    GameIdentity,
    SearchObservation,
    SearchStopReason,
    SearchVisitCounts,
    TerminationReason,
)
from src.training.checkpoint import CheckpointReference


def _checkpoint(path: Path, generation: int) -> CheckpointReference:
    return CheckpointReference(
        generation=generation,
        manifest_path=path / 'checkpoint.json',
        model_path=path / 'model.pt',
        optimizer_path=path / 'optimizer.pt',
        inference_model_path=path / 'model-inference.pt',
        inference_model_sha256='0' * 64,
    )


def _observation(ply: int, model_generation: int) -> SearchObservation:
    return SearchObservation(
        ply=ply,
        model_generation=model_generation,
        policy_target_visits=SearchVisitCounts(action_ids=(0, 1), visit_counts=(8, 2)),
        root_value=0.1,
        highest_visited_child_action_id=0,
        highest_visited_child_visit_count=8,
        highest_visited_child_q=0.1,
        selected_action_id=0,
        sample_weight=1.0,
        baseline_visits=10,
        network_root_value=0.0,
        policy_correction=0.0,
        value_correction=0.1,
        search_budget_logit=0.0,
        predicted_search_budget=0.5,
        assigned_additional_visits=10,
        parallel_searches=1,
        spend_residual=0,
        starting_visits=0,
        final_visits=10,
        stop_reason=SearchStopReason.FIXED_LIMIT,
    )


def _game(game_number: int, observation_generations: tuple[int, ...]) -> ReplayShardGameMetadata:
    identity = GameIdentity(
        worker_id=1,
        process_instance_id=UUID('778a33fa-a5d6-4861-9098-ccf8945a2b61'),
        game_number=game_number,
    )
    observations = tuple(
        _observation(ply, model_generation) for ply, model_generation in enumerate(observation_generations)
    )
    return ReplayShardGameMetadata(
        source=ReplayShardSourceGame(identity=identity, counter=game_number),
        created_at_seconds=100.0,
        generation_seconds=5.0,
        action_ids=(0,) * len(observations),
        row_start=0,
        row_count=len(observations),
        length_plies=len(observations),
        termination_reason=TerminationReason.NATURAL,
        is_resignation_continuation=False,
        resignation_threshold=None,
        final_wdl=WdlTarget(win=1.0, draw=0.0, loss=0.0),
        observations=observations,
        policies_truncated=0,
        retained_visit_mass=10 * len(observations),
        discarded_visit_mass=0,
    )


def _label_games(
    games: tuple[ReplayShardGameMetadata, ...],
) -> tuple[tuple[ReplayLabelGameLocator, ...], dict[int, ReplaySample]]:
    locators = []
    samples: dict[int, ReplaySample] = {}
    first_absolute_row = 100
    for game in games:
        locators.append(
            ReplayLabelGameLocator(
                identity=game.source.identity,
                action_ids=game.action_ids,
                observation_plies=tuple(observation.ply for observation in game.observations),
                first_absolute_replay_row=first_absolute_row,
            )
        )
        for observation_index in range(len(game.observations)):
            samples[first_absolute_row + observation_index] = _sample(game, observation_index)
        first_absolute_row += len(game.observations)
    return tuple(locators), samples


def test_generation_source_samples_exact_floor_two_percent_deterministically(tmp_path: Path) -> None:
    games = (_game(1, (4,) * 51), _game(2, (4,) * 98))
    label_games, samples = _label_games(games)

    first = build_generation_source(
        4, label_games, _checkpoint(tmp_path, 4), 600, 123, Decimal('0.02'), samples.__getitem__
    )
    second = build_generation_source(
        4, label_games, _checkpoint(tmp_path, 4), 600, 123, Decimal('0.02'), samples.__getitem__
    )

    assert first is not None
    assert second is not None
    assert first.population_position_count == 149
    assert len(first.selected_positions) == 2
    assert first.selected_positions == second.selected_positions
    assert first.deep_visit_limit == 4_800


def test_cross_checkpoint_game_keeps_every_cohort_observation(tmp_path: Path) -> None:
    game = _game(7, (3, 3, 4, 4, 5))
    label_games, samples = _label_games((game,))

    source = build_generation_source(4, label_games, _checkpoint(tmp_path, 4), 300, 9, Decimal(1), samples.__getitem__)

    assert source is not None
    assert source.population_position_count == 5
    assert tuple(sorted(position.identity.ply for position in source.selected_positions)) == (0, 1, 2, 3, 4)
    assert {position.replay.source_model_generation for position in source.selected_positions} == {
        3,
        4,
        5,
    }
    assert all(position.identity.source_generation == 4 for position in source.selected_positions)


def test_generation_source_is_compact_durable_and_materializes_only_selected_positions(tmp_path: Path) -> None:
    game = _game(9, (4,) * 50)
    label_games, samples = _label_games((game,))
    materialized_rows: list[int] = []

    def recording_sample_provider(absolute_replay_row: int) -> ReplaySample:
        materialized_rows.append(absolute_replay_row)
        return samples[absolute_replay_row]

    source = build_generation_source(
        4,
        label_games,
        _checkpoint(tmp_path, 4),
        600,
        123,
        Decimal('0.4'),
        recording_sample_provider,
    )

    assert source is not None
    assert len(source.selected_positions) == 20
    assert len(materialized_rows) == 20
    assert sorted(row - 100 for row in materialized_rows) == sorted(
        position.identity.ply for position in source.selected_positions
    )
    path = tmp_path / 'source.json'
    write_persisted_model(path, source)
    restored = load_persisted_model(path, LabelGenerationSource)
    assert restored == source
    assert tuple(position.replay.replay_sample() for position in restored.selected_positions) == tuple(
        _sample(game, position.identity.ply) for position in restored.selected_positions
    )

    repeated_game_payload = {
        'selected_positions': [
            {
                'identity': position.identity.model_dump(mode='json'),
                'game': game.model_dump(mode='json'),
                'observation_index': position.identity.ply,
            }
            for position in source.selected_positions
        ]
    }
    repeated_size = len(json.dumps(repeated_game_payload, separators=(',', ':')).encode())
    assert path.stat().st_size < repeated_size // 20


def _sample(game: ReplayShardGameMetadata, observation_index: int) -> ReplaySample:
    observation = game.observations[observation_index]
    return ReplaySample(
        encoded_state=PackedPlanePayload(bytes(8)),
        policy=SparsePolicyTarget(
            visits=SearchVisitCounts(action_ids=(0, 1), visit_counts=(5, 5)),
            legal_action_ids=(0, 1),
        ),
        wdl_target=WdlTarget(win=1.0, draw=0.0, loss=0.0),
        root_value=0.0,
        auxiliary_targets=(IneligibleSearchBudgetTarget(),),
        sample_weight=1.0,
        source_model_generation=observation.model_generation,
        source_created_at_seconds=100.0,
    )


def test_finalization_ranks_all_generation_kl_values_and_writes_deep_policy(tmp_path: Path) -> None:
    game = _game(8, (4, 4, 4))
    positions = tuple(
        LabelPositionSource(
            identity=LabelPositionIdentity(
                source_generation=4,
                game_identity=game.source.identity.archive_key,
                ply=index,
            ),
            action_prefix=game.action_ids[:index],
            replay=LabelReplaySampleSource.from_replay_sample(_sample(game, index)),
        )
        for index in range(3)
    )
    source = LabelGenerationSource(
        source_generation=4,
        population_position_count=150,
        baseline_new_visits=10,
        checkpoint=_checkpoint(tmp_path, 4),
        selected_positions=positions,
    )
    predictions = {
        position.identity: PredictionRecord(
            identity=position.identity,
            search_budget_logit=0.0,
            predicted_quantile=0.5,
        )
        for position in positions
    }
    allocations = candidate_allocations(source, predictions, initial_calibration_state('a' * 64), 1.1)
    checkpoints = checkpoint_visits_by_position(source, allocations)
    records = tuple(
        DeepSearchRecord(
            identity=position.identity,
            checkpoints=tuple(
                PolicyCheckpointRecord(
                    visits=visits,
                    policy_target_visits=SearchVisitCounts(
                        action_ids=(0, 1),
                        visit_counts=(5, 5) if index == 1 else (9, 1),
                    ),
                )
                for visits in checkpoints[position.identity]
            ),
            final_policy_target_visits=SearchVisitCounts(action_ids=(0, 1), visit_counts=(40, 40)),
            final_root_value=0.25,
            starting_visits=0,
            final_visits=source.deep_visit_limit,
        )
        for index, position in enumerate(positions)
    )
    artifact = DeepSearchShardArtifact(
        source_generation=4,
        shard_index=0,
        checkpoint_sha256=source.checkpoint.inference_model_sha256,
        records=records,
    )

    finalized = finalize_generation(
        source,
        predictions,
        allocations,
        (artifact,),
        action_size=2,
        maximum_policy_entries=2,
    )

    targets = tuple(
        target
        for sample in finalized.replay_samples
        for target in sample.auxiliary_targets
        if isinstance(target, EligibleSearchBudgetTarget)
    )
    assert tuple(target.normalized_target for target in targets) == (0.75, 0.0, 0.75)
    assert all(sample.policy.visits.visit_counts == (40, 40) for sample in finalized.replay_samples)
    assert all(sample.root_value == 0.25 for sample in finalized.replay_samples)
    assert finalized.validation_diagnostics.exact_spend_residual is None
    assert finalized.evidence.validated_curve is None
    assert sum(diagnostic.sample_count for diagnostic in finalized.bucket_diagnostics) == 3
    assert finalized.bucket_diagnostics[5].generation_marginal_utility == 0.0
    assert finalized.target_distribution.variance > 0.0
    assert sum(finalized.target_distribution.histogram_counts) == 3

    lower_identity = CurveAllocationIdentity(CurveAllocationPurpose.PROBE_LOWER, 5)
    upper_identity = CurveAllocationIdentity(CurveAllocationPurpose.PROBE_UPPER, 5)
    lower = next(allocation for allocation in allocations if allocation.identity == lower_identity)
    upper = next(allocation for allocation in allocations if allocation.identity == upper_identity)
    inverted_upper = replace(
        upper,
        budgets=(
            replace(
                upper.budgets[0],
                assigned_new_visits=lower.budgets[0].assigned_new_visits - 1,
            ),
            *upper.budgets[1:],
        ),
    )
    inverted_allocations = tuple(
        inverted_upper if allocation.identity == upper_identity else allocation for allocation in allocations
    )
    rounded = finalize_generation(
        source,
        predictions,
        inverted_allocations,
        (artifact,),
        action_size=2,
        maximum_policy_entries=2,
    )
    assert rounded.bucket_diagnostics[5].checkpoint_deduplication_count == 3
    assert rounded.bucket_diagnostics[5].sample_count == 2

    materially_inverted_upper = replace(
        upper,
        budgets=(
            replace(
                upper.budgets[0],
                assigned_new_visits=lower.budgets[0].assigned_new_visits - 2,
            ),
            *upper.budgets[1:],
        ),
    )
    with pytest.raises(ValueError, match='materially below'):
        finalize_generation(
            source,
            predictions,
            tuple(
                materially_inverted_upper if allocation.identity == upper_identity else allocation
                for allocation in allocations
            ),
            (artifact,),
            action_size=2,
            maximum_policy_entries=2,
        )

    invalid_positions = tuple(
        position.model_copy(
            update={
                'replay': LabelReplaySampleSource.from_replay_sample(
                    replace(
                        position.replay.replay_sample(),
                        auxiliary_targets=(IneligibleSearchBudgetTarget(), IneligibleSearchBudgetTarget()),
                    )
                )
            }
        )
        for position in positions
    )
    with pytest.raises(ValueError, match='exactly one'):
        finalize_generation(
            source.model_copy(update={'selected_positions': invalid_positions}),
            predictions,
            allocations,
            (artifact,),
            action_size=2,
            maximum_policy_entries=2,
        )
