from __future__ import annotations

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
from src.replay.shard import ReplayShardGameMetadata, ReplayShardSourceGame
from src.search_budget.labeling import (
    DeepSearchRecord,
    DeepSearchShardArtifact,
    LabelGenerationSource,
    LabelPositionSource,
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


def test_generation_source_samples_exact_floor_two_percent_deterministically(tmp_path: Path) -> None:
    games = (_game(1, (4,) * 51), _game(2, (4,) * 98))

    first = build_generation_source(4, games, _checkpoint(tmp_path, 4), 600, 123, Decimal('0.02'))
    second = build_generation_source(4, games, _checkpoint(tmp_path, 4), 600, 123, Decimal('0.02'))

    assert first is not None
    assert second is not None
    assert first.population_position_count == 149
    assert len(first.selected_positions) == 2
    assert first.selected_positions == second.selected_positions
    assert first.deep_visit_limit == 4_800


def test_cross_checkpoint_game_keeps_every_cohort_observation(tmp_path: Path) -> None:
    game = _game(7, (3, 3, 4, 4, 5))

    source = build_generation_source(4, (game,), _checkpoint(tmp_path, 4), 300, 9, Decimal(1))

    assert source is not None
    assert source.population_position_count == 5
    assert tuple(sorted(position.observation_index for position in source.selected_positions)) == (0, 1, 2, 3, 4)
    assert {
        position.game.observations[position.observation_index].model_generation
        for position in source.selected_positions
    } == {
        3,
        4,
        5,
    }
    assert all(position.identity.source_generation == 4 for position in source.selected_positions)


def _base_sample(source: LabelPositionSource) -> ReplaySample:
    del source
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
        source_model_generation=4,
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
            game=game,
            observation_index=index,
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
    allocations = candidate_allocations(source, predictions)
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
        sample_provider=_base_sample,
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
    assert all(diagnostic.exact_spend_residual == 0 for diagnostic in finalized.candidate_diagnostics)
    assert all(diagnostic.assigned_new_visits_variance == 0.0 for diagnostic in finalized.candidate_diagnostics)
    assert all(diagnostic.mean_kl_from_deep >= 0.0 for diagnostic in finalized.candidate_diagnostics)
    assert finalized.target_distribution.variance > 0.0
    assert sum(finalized.target_distribution.histogram_counts) == 3

    def duplicate_target_provider(position: LabelPositionSource) -> ReplaySample:
        base = _base_sample(position)
        return replace(
            base,
            auxiliary_targets=(IneligibleSearchBudgetTarget(), IneligibleSearchBudgetTarget()),
        )

    with pytest.raises(ValueError, match='exactly one'):
        finalize_generation(
            source,
            predictions,
            allocations,
            (artifact,),
            action_size=2,
            maximum_policy_entries=2,
            sample_provider=duplicate_target_provider,
        )
