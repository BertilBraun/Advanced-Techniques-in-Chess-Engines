from pathlib import Path

from src.evaluation.configuration import StockfishEngineConfiguration
from src.evaluation.contracts import CandidateOutcome, EvaluationGameResult, EvaluationTerminationReason
from src.experiment.configuration import load_chess_experiment_configuration
from src.self_play.configuration import MonteCarloGraphSearchConfiguration
from tools.run_stockfish_gauntlet import (
    FixedModelSearchBudget,
    GauntletShardResult,
    PrefixOpeningSelection,
    SeededOpeningSelection,
    TimedModelSearchBudget,
    TimedMoveMeasurements,
    _combine_timed_measurements,
    _pair_shards,
    _search_configuration,
    _select_opening_indices,
    _shift_game_indices,
    _stockfish_configuration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_PATH = PROJECT_ROOT / 'py/configs/production/vast-chess-8gpu-1d-r3.yaml'


def test_fixed_budget_reproduces_production_evaluation_search() -> None:
    budget = FixedModelSearchBudget(
        searches_per_move=64,
        parallel_searches=1,
        inference_workers=1,
        inference_batch_size=64,
        outstanding_batches_per_worker=1,
    )

    search = _search_configuration(budget)

    assert search.searches_per_move == 64
    assert search.parallel_searches == 1
    assert search.exploration_constant == 1.0
    assert search.inference.inference_workers == 1
    assert search.inference.inference_batch_size == 64
    assert search.inference.outstanding_batches_per_worker == 1


def test_timed_budget_builds_only_internal_validation_search() -> None:
    budget = TimedModelSearchBudget(
        seconds_per_move=5,
        parallel_searches=64,
        inference_workers=2,
        inference_batch_size=64,
        outstanding_batches_per_worker=2,
    )

    search = _search_configuration(budget)

    assert search.searches_per_move == 65
    assert search.parallel_searches == 64


def test_graph_strength_budget_preserves_the_algorithm_discriminator() -> None:
    algorithm = MonteCarloGraphSearchConfiguration(transposition_value_threshold=0.02)
    budget = FixedModelSearchBudget(
        searches_per_move=400,
        parallel_searches=4,
        inference_workers=1,
        inference_batch_size=64,
        outstanding_batches_per_worker=1,
        algorithm=algorithm,
    )

    assert _search_configuration(budget).algorithm == algorithm


def test_pair_shards_are_balanced_and_contiguous() -> None:
    shards = _pair_shards(50, tuple(range(8)))

    assert shards == (
        (0, 0, 7),
        (1, 7, 7),
        (2, 14, 6),
        (3, 20, 6),
        (4, 26, 6),
        (5, 32, 6),
        (6, 38, 6),
        (7, 44, 6),
    )


def test_prefix_opening_selection_preserves_manifest_order() -> None:
    assert _select_opening_indices(200, 10, PrefixOpeningSelection()) == tuple(range(10))


def test_seeded_opening_selection_is_reproducible_and_spans_manifest() -> None:
    selected = _select_opening_indices(200, 10, SeededOpeningSelection(random_seed=20260815))

    assert selected == _select_opening_indices(200, 10, SeededOpeningSelection(random_seed=20260815))
    assert len(selected) == len(set(selected)) == 10
    assert min(selected) >= 0
    assert max(selected) < 200
    assert selected != tuple(range(10))


def test_shift_game_indices_preserves_game_evidence() -> None:
    game = EvaluationGameResult(
        game_index=1,
        pair_index=0,
        opening_id='opening',
        candidate_player='second',
        pair_seed=7,
        initial_action_ids=(1, 2, 3, 4),
        played_action_ids=(5, 6),
        outcome=CandidateOutcome.WIN,
        termination_reason=EvaluationTerminationReason.NATURAL,
        plies=2,
        duration_seconds=1.0,
    )

    shifted = _shift_game_indices(game, first_pair_index=7)

    assert shifted.game_index == 15
    assert shifted.pair_index == 7
    assert shifted.opening_id == game.opening_id
    assert shifted.played_action_ids == game.played_action_ids


def test_timed_measurements_combine_with_move_weighting() -> None:
    first = TimedMoveMeasurements(
        move_count=2,
        total_searches=200,
        minimum_searches=90,
        maximum_searches=110,
        mean_searches=100.0,
        total_elapsed_milliseconds=2_020,
        minimum_elapsed_milliseconds=1_000,
        maximum_elapsed_milliseconds=1_020,
        mean_elapsed_milliseconds=1_010.0,
    )
    second = TimedMoveMeasurements(
        move_count=1,
        total_searches=130,
        minimum_searches=130,
        maximum_searches=130,
        mean_searches=130.0,
        total_elapsed_milliseconds=1_010,
        minimum_elapsed_milliseconds=1_010,
        maximum_elapsed_milliseconds=1_010,
        mean_elapsed_milliseconds=1_010.0,
    )
    game = EvaluationGameResult(
        game_index=0,
        pair_index=0,
        opening_id='opening',
        candidate_player='first',
        pair_seed=7,
        initial_action_ids=(1, 2, 3, 4),
        played_action_ids=(5, 6),
        outcome=CandidateOutcome.DRAW,
        termination_reason=EvaluationTerminationReason.NATURAL,
        plies=2,
        duration_seconds=1.0,
    )
    shards = (
        GauntletShardResult(
            shard_id=0,
            device_id=0,
            first_pair_index=0,
            pair_count=1,
            stockfish_identity='Stockfish 13',
            games=(game, game.model_copy(update={'game_index': 1})),
            timed_move_measurements=first,
            duration_seconds=1.0,
        ),
        GauntletShardResult(
            shard_id=1,
            device_id=1,
            first_pair_index=1,
            pair_count=1,
            stockfish_identity='Stockfish 13',
            games=(
                game.model_copy(update={'game_index': 2, 'pair_index': 1}),
                game.model_copy(update={'game_index': 3, 'pair_index': 1}),
            ),
            timed_move_measurements=second,
            duration_seconds=1.0,
        ),
    )

    combined = _combine_timed_measurements(shards)

    assert combined is not None
    assert combined.move_count == 3
    assert combined.total_searches == 330
    assert combined.mean_searches == 110.0
    assert combined.minimum_searches == 90
    assert combined.maximum_searches == 130


def test_gauntlet_preserves_stockfish_resources_and_overrides_nodes(tmp_path: Path) -> None:
    configuration = load_chess_experiment_configuration(EXPERIMENT_PATH)
    executable = tmp_path / 'stockfish-13'

    engine = _stockfish_configuration(configuration, executable, match_nodes=5_000)

    assert isinstance(engine, StockfishEngineConfiguration)
    assert engine.executable_path == str(executable.resolve())
    assert engine.match_nodes == 5_000
    assert engine.threads == 1
    assert engine.hash_mib == 1_024
