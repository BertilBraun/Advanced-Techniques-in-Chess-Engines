from __future__ import annotations

import json
import math
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import numpy as np
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
from src.search_budget.artifacts import load_persisted_model, write_persisted_model
from src.search_budget.labeling import (
    DeepSearchRecord,
    DeepSearchShardArtifact,
    LabelGenerationSource,
    LabelPositionSource,
    LabelReplaySampleSource,
    PolicyCheckpointRecord,
    PredictionRecord,
    build_generation_source,
    finalize_generation,
)
from src.search_budget.policy import (
    BASELINE_CURVE_INDEX,
    BUDGET_CURVE_POINTS,
    SearchBudgetPolicy,
    grid_visit_counts,
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
        predicted_baseline_log_kl=0.0,
        selected_budget_index=-1,
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
    assert first.checkpoint_visits == (75, 120, 200, 300, 400, 600, 900, 1200, 1800, 2400)


def test_generation_source_records_each_selected_absolute_replay_row(tmp_path: Path) -> None:
    game = _game(9, (4,) * 50)
    label_games, samples = _label_games((game,))
    source = build_generation_source(
        4, label_games, _checkpoint(tmp_path, 4), 600, 123, Decimal(1), samples.__getitem__
    )
    assert source is not None
    assert tuple(position.absolute_replay_row - 100 for position in source.selected_positions) == tuple(
        position.identity.ply for position in source.selected_positions
    )


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


def _finalization_source(tmp_path: Path) -> LabelGenerationSource:
    game = _game(8, (4, 4, 4))
    positions = tuple(
        LabelPositionSource(
            identity=LabelPositionIdentity(
                source_generation=4,
                game_identity=game.source.identity.archive_key,
                ply=index,
            ),
            action_prefix=game.action_ids[:index],
            absolute_replay_row=100 + index,
            replay=LabelReplaySampleSource.from_replay_sample(_sample(game, index)),
        )
        for index in range(3)
    )
    return LabelGenerationSource(
        source_generation=4,
        population_position_count=150,
        baseline_new_visits=10,
        checkpoint=_checkpoint(tmp_path, 4),
        selected_positions=positions,
    )


def _deep_artifact(source: LabelGenerationSource) -> DeepSearchShardArtifact:
    # Every checkpoint below the deepest keeps a lopsided policy; the deepest checkpoint and the
    # final policy agree, so the raw KL curve is flat until it drops to zero at the last grid point.
    records = tuple(
        DeepSearchRecord(
            identity=position.identity,
            checkpoints=tuple(
                PolicyCheckpointRecord(
                    visits=visits,
                    root_value=0.05 if visits < source.deep_visit_limit // 2 else 0.25,
                    policy_target_visits=SearchVisitCounts(
                        action_ids=(0, 1),
                        visit_counts=(40, 40) if visits == source.checkpoint_visits[-1] else (9, 1),
                    ),
                )
                for visits in source.checkpoint_visits
            ),
            final_policy_target_visits=SearchVisitCounts(action_ids=(0, 1), visit_counts=(40, 40)),
            final_root_value=0.25,
            starting_visits=0,
            final_visits=source.deep_visit_limit,
        )
        for position in source.selected_positions
    )
    return DeepSearchShardArtifact(
        source_generation=source.source_generation,
        shard_index=0,
        checkpoint_sha256=source.checkpoint.inference_model_sha256,
        records=records,
    )


def _predictions(
    source: LabelGenerationSource,
    curves: tuple[tuple[float, ...], ...],
) -> dict[LabelPositionIdentity, PredictionRecord]:
    return {
        position.identity: PredictionRecord(identity=position.identity, predicted_curve=curve)
        for position, curve in zip(source.selected_positions, curves, strict=True)
    }


def _policy() -> SearchBudgetPolicy:
    return SearchBudgetPolicy(
        sigma=(1.0,) * BUDGET_CURVE_POINTS,
        log_tau=math.log(0.1),
        selection_threshold=0.8,
        apply_learned=True,
    )


def test_finalization_writes_log_kl_curve_targets_and_the_deep_policy(tmp_path: Path) -> None:
    source = _finalization_source(tmp_path)
    predictions = _predictions(source, ((5.0,) * BUDGET_CURVE_POINTS,) * 3)

    finalized = finalize_generation(
        source,
        predictions,
        _policy(),
        (_deep_artifact(source),),
        action_size=2,
        maximum_policy_entries=2,
    )

    targets = tuple(
        target
        for sample in finalized.replay_samples
        for target in sample.auxiliary_targets
        if isinstance(target, EligibleSearchBudgetTarget)
    )
    assert len(targets) == 3
    shallow_kl = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    expected_curve = (math.log(shallow_kl + 1e-6),) * (BUDGET_CURVE_POINTS - 1) + (math.log(1e-6),)
    for target in targets:
        assert target.curve == pytest.approx(expected_curve)
        assert target.raw_kl == pytest.approx(shallow_kl)
    assert all(sample.policy.visits.visit_counts == (40, 40) for sample in finalized.replay_samples)
    assert all(sample.root_value == 0.25 for sample in finalized.replay_samples)


def test_finalization_measures_shadow_gain_and_selection_under_the_working_policy(tmp_path: Path) -> None:
    source = _finalization_source(tmp_path)
    # Two positions predict confidently cheap curves and select the cheapest grid point; one
    # predicts a hard curve and falls back to the deepest grid point.
    cheap = (-10.0,) * BUDGET_CURVE_POINTS
    hard = (5.0,) * BUDGET_CURVE_POINTS
    predictions = _predictions(source, (cheap, hard, cheap))

    finalized = finalize_generation(
        source,
        predictions,
        _policy(),
        (_deep_artifact(source),),
        action_size=2,
        maximum_policy_entries=2,
    )

    grid = grid_visit_counts(source.baseline_new_visits)
    evidence = finalized.evidence
    assert evidence.selected_index_counts == (2, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    assert evidence.realized_mean_multiple == pytest.approx((0.125 + 0.125 + 4.0) / 3)
    assert evidence.realized_mean_assigned_visits == pytest.approx((grid[0] + grid[0] + grid[-1]) / 3)
    assert evidence.flat_mean_assigned_visits == pytest.approx(10.0)
    # The deepest checkpoint matches the deep policy while cheap checkpoints do not, so the one
    # deep selection gains exactly one position's baseline KL over flat allocation.
    shallow_kl = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    assert evidence.generation_gain == pytest.approx(shallow_kl / 3)
    assert evidence.mean_absolute_curve_error == pytest.approx(
        tuple(
            (
                2 * abs(-10.0 - math.log(shallow_kl + 1e-6)) + abs(5.0 - math.log(shallow_kl + 1e-6))
                if index < BUDGET_CURVE_POINTS - 1
                else 2 * abs(-10.0 - math.log(1e-6)) + abs(5.0 - math.log(1e-6))
            )
            / 3
            for index in range(BUDGET_CURVE_POINTS)
        )
    )


def test_finalization_appends_one_analysis_record_per_labelled_position(tmp_path: Path) -> None:
    source = _finalization_source(tmp_path)
    cheap = (-10.0,) * BUDGET_CURVE_POINTS
    hard = (5.0,) * BUDGET_CURVE_POINTS
    predictions = _predictions(source, (cheap, hard, cheap))

    finalized = finalize_generation(
        source,
        predictions,
        _policy(),
        (_deep_artifact(source),),
        action_size=2,
        maximum_policy_entries=2,
    )

    records = finalized.analysis_records
    assert records.shape == (3,)
    assert list(records['source_generation']) == [4, 4, 4]
    assert list(records['ply']) == [0, 1, 2]
    assert list(records['first_absolute_replay_row']) == [100, 101, 102]
    assert list(records['baseline_visits']) == [10, 10, 10]
    assert list(records['selected_index']) == [0, 9, 0]
    grid = grid_visit_counts(10)
    assert list(records['assigned_visits']) == [grid[0], grid[-1], grid[0]]
    shallow_kl = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    assert records['policy_kl'][0][BASELINE_CURVE_INDEX] == pytest.approx(shallow_kl, rel=1e-6)
    assert records['deep_half_kl'][0] == pytest.approx(0.0, abs=1e-9)
    assert records['predicted_curve'][1][0] == pytest.approx(5.0)
    # Checkpoints below half depth carry root value 0.05 against the deep 0.25.
    assert records['value_error'][0][0] == pytest.approx(0.2, rel=1e-5)
    assert records['value_error'][0][BUDGET_CURVE_POINTS - 1] == pytest.approx(0.0, abs=1e-9)
    assert records['top_visit_share'][0] == pytest.approx(0.9)
    assert records['policy_entropy'][0] == pytest.approx(-(0.9 * math.log(0.9) + 0.1 * math.log(0.1)), rel=1e-5)
    assert records.dtype.itemsize <= 200


def test_finalization_requires_exactly_one_search_budget_slot(tmp_path: Path) -> None:
    source = _finalization_source(tmp_path)
    predictions = _predictions(source, ((0.0,) * BUDGET_CURVE_POINTS,) * 3)
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
        for position in source.selected_positions
    )
    with pytest.raises(ValueError, match='exactly one'):
        finalize_generation(
            source.model_copy(update={'selected_positions': invalid_positions}),
            predictions,
            _policy(),
            (_deep_artifact(source),),
            action_size=2,
            maximum_policy_entries=2,
        )


def test_finalization_rejects_a_missing_grid_checkpoint(tmp_path: Path) -> None:
    source = _finalization_source(tmp_path)
    predictions = _predictions(source, ((0.0,) * BUDGET_CURVE_POINTS,) * 3)
    artifact = _deep_artifact(source)
    truncated = DeepSearchShardArtifact(
        source_generation=artifact.source_generation,
        shard_index=artifact.shard_index,
        checkpoint_sha256=artifact.checkpoint_sha256,
        records=tuple(record.model_copy(update={'checkpoints': record.checkpoints[1:]}) for record in artifact.records),
    )
    with pytest.raises(ValueError, match='missing required checkpoint'):
        finalize_generation(
            source,
            predictions,
            _policy(),
            (truncated,),
            action_size=2,
            maximum_policy_entries=2,
        )


def test_analysis_record_dtype_is_fixed_width_and_reasonably_small() -> None:
    from src.search_budget.analysis_log import ANALYSIS_RECORD_DTYPE

    assert ANALYSIS_RECORD_DTYPE.itemsize <= 200
    assert ANALYSIS_RECORD_DTYPE['policy_kl'].shape == (BUDGET_CURVE_POINTS,)
    assert ANALYSIS_RECORD_DTYPE['value_error'].shape == (BUDGET_CURVE_POINTS,)
    assert ANALYSIS_RECORD_DTYPE['predicted_curve'].shape == (BUDGET_CURVE_POINTS,)
    assert np.dtype(ANALYSIS_RECORD_DTYPE) is not None
