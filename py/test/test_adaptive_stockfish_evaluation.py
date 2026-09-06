from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path

import pytest
from src.evaluation.configuration import (
    StockfishAdaptiveNodesEvaluationDefinition,
)
from src.evaluation.contracts import (
    EVALUATION_JOB_ADAPTER,
    CandidateOutcome,
    EvaluationGameResult,
    EvaluationTerminationReason,
    FixedDatasetEvaluationJob,
    FixedDatasetEvaluationResult,
    MatchAggregate,
    MatchEvaluationJob,
    MatchEvaluationResult,
    StockfishFixedNodesOpponent,
)
from src.evaluation.manager import EvaluationManager, EvaluationManagerState
from src.evaluation.process import write_evaluation_result
from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.training.checkpoint import CheckpointReference
from test_helpers.checkpoints import checkpoint_reference
from test_helpers.configuration_paths import REPOSITORY_CONFIG_DIRECTORY, TEST_CONFIG_DIRECTORY


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeProcess:
    def __init__(
        self,
        target: Callable[[str, str], None],
        args: tuple[str, str],
        name: str,
    ) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.exitcode: int | None = None
        self.pid: int | None = None
        self.started = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started and self.exitcode is None

    def terminate(self) -> None:
        self.exitcode = -15

    def kill(self) -> None:
        self.exitcode = -9

    def join(self, timeout: float | None = None) -> None:
        if self.exitcode is None:
            self.exitcode = 0


class FakeProcessContext:
    def __init__(self) -> None:
        self.processes: list[FakeProcess] = []

    def Process(
        self,
        target: Callable[[str, str], None],
        args: tuple[str, str],
        name: str,
    ) -> FakeProcess:
        process = FakeProcess(target, args, name)
        self.processes.append(process)
        return process


def _checkpoint(run_path: Path, generation: int) -> CheckpointReference:
    return checkpoint_reference(run_path, generation)


def _adaptive_experiment(run_path: Path) -> ChessExperimentConfiguration:
    loaded = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    assert isinstance(loaded, ChessExperimentConfiguration)
    stockfish = next(definition for definition in loaded.evaluation.definitions if definition.kind == 'stockfish')
    fixed_dataset = next(
        definition for definition in loaded.evaluation.definitions if definition.kind == 'fixed_dataset'
    )
    searched = StockfishAdaptiveNodesEvaluationDefinition(
        kind='stockfish_adaptive_nodes',
        definition_id='stockfish-searched',
        node_ladder=(30, 100, 300, 1_000),
        initial_nodes=30,
        retreat_score_threshold=0.30,
        advance_score_threshold=0.70,
        engine_executable_path='engines/stockfish-13',
        opening_pair_count=50,
        maximum_game_plies=300,
        first_generation=0,
        final_generation=None,
        search=stockfish.search,
    )
    policy_only = StockfishAdaptiveNodesEvaluationDefinition(
        kind='stockfish_adaptive_nodes',
        definition_id='stockfish-policy-only',
        node_ladder=(30, 100, 300, 1_000),
        initial_nodes=30,
        retreat_score_threshold=0.30,
        advance_score_threshold=0.70,
        engine_executable_path='engines/stockfish-13',
        opening_pair_count=50,
        maximum_game_plies=300,
        first_generation=0,
        final_generation=None,
        search=stockfish.search.model_copy(update={'searches_per_move': 1, 'parallel_searches': 1}),
    )
    training = loaded.training.model_copy(update={'save_path': str(run_path)})
    evaluation = loaded.evaluation.model_copy(
        update={
            'cadence_seconds': 20,
            'maximum_concurrent_jobs': 3,
            'definitions': (fixed_dataset, searched, policy_only),
        }
    )
    return loaded.model_copy(update={'training': training, 'evaluation': evaluation})


def _games(score: float) -> tuple[EvaluationGameResult, ...]:
    wins = round(score * 100)
    outcomes = (CandidateOutcome.WIN,) * wins + (CandidateOutcome.LOSS,) * (100 - wins)
    return tuple(
        EvaluationGameResult(
            game_index=game_index,
            pair_index=game_index // 2,
            opening_id=f'opening-{game_index // 2}',
            candidate_player='first' if game_index % 2 == 0 else 'second',
            pair_seed=game_index // 2,
            initial_action_ids=(),
            played_action_ids=(),
            outcome=outcome,
            termination_reason=EvaluationTerminationReason.NATURAL,
            plies=0,
            duration_seconds=1.0,
        )
        for game_index, outcome in enumerate(outcomes)
    )


def _write_match_result(job: MatchEvaluationJob, score: float) -> None:
    games = _games(score)
    wins = sum(game.outcome is CandidateOutcome.WIN for game in games)
    losses = len(games) - wins
    write_evaluation_result(
        MatchEvaluationResult(
            kind='match',
            job=job,
            games=games,
            aggregate=MatchAggregate(
                wins=wins,
                draws=0,
                losses=losses,
                score=score,
                first_player_score=score,
                second_player_score=score,
                pair_count=50,
                score_confidence_low=max(0.0, score - 0.1),
                score_confidence_high=min(1.0, score + 0.1),
            ),
            duration_seconds=1.0,
        ),
        job.result_path,
    )


def _complete_boundary(
    manager: EvaluationManager,
    context: FakeProcessContext,
    jobs: tuple[FixedDatasetEvaluationJob | MatchEvaluationJob, ...],
    searched_score: float,
    policy_score: float = 0.5,
) -> None:
    for job in jobs:
        if isinstance(job, FixedDatasetEvaluationJob):
            write_evaluation_result(
                FixedDatasetEvaluationResult(
                    kind='fixed_dataset',
                    job=job,
                    position_count=500,
                    source_game_count=25,
                    top_action_accuracy=0.25,
                    policy_cross_entropy=2.0,
                    duration_seconds=1.0,
                ),
                job.result_path,
            )
        else:
            score = policy_score if job.definition.definition_id == 'stockfish-policy-only' else searched_score
            _write_match_result(job, score)
    scheduled_ids = {job.job_id for job in jobs}
    for process in context.processes:
        process_job = EVALUATION_JOB_ADAPTER.validate_json(process.args[1])
        if process_job.job_id in scheduled_ids:
            process.exitcode = 0
    assert len(manager.collect_completed_jobs()) == len(jobs)


def _adaptive_jobs(
    jobs: tuple[FixedDatasetEvaluationJob | MatchEvaluationJob, ...],
) -> tuple[MatchEvaluationJob, ...]:
    return tuple(
        job
        for job in jobs
        if isinstance(job, MatchEvaluationJob)
        and isinstance(job.definition, StockfishAdaptiveNodesEvaluationDefinition)
    )


def _nodes(job: MatchEvaluationJob) -> int:
    assert isinstance(job.opponent, StockfishFixedNodesOpponent)
    return job.opponent.nodes


def test_v33_configures_only_the_dataset_and_two_adaptive_matches() -> None:
    experiment = load_experiment_configuration(
        REPOSITORY_CONFIG_DIRECTORY / 'production' / 'vast-chess-8gpu-integrated-v33.yaml'
    )
    assert isinstance(experiment, ChessExperimentConfiguration)

    assert tuple(definition.definition_id for definition in experiment.evaluation.definitions) == (
        'fixed-dataset',
        'stockfish-searched',
        'stockfish-policy-only',
    )
    adaptive = tuple(
        definition
        for definition in experiment.evaluation.definitions
        if isinstance(definition, StockfishAdaptiveNodesEvaluationDefinition)
    )
    assert tuple(definition.search.searches_per_move for definition in adaptive) == (64, 1)
    assert all(definition.search.parallel_searches == 1 for definition in adaptive)
    assert all(definition.opening_pair_count == 50 for definition in adaptive)
    assert all(definition.node_ladder == (30, 100, 300, 1_000, 2_000, 3_000, 5_000, 10_000) for definition in adaptive)


def test_adaptive_suite_starts_at_configured_rungs_and_schedules_one_match_per_mode(tmp_path: Path) -> None:
    experiment = _adaptive_experiment(tmp_path)
    clock = FakeClock()
    context = FakeProcessContext()
    manager = EvaluationManager(experiment, _checkpoint(tmp_path, 0), clock, context)

    clock.now = 21.0
    jobs = manager.schedule_due_jobs(_checkpoint(tmp_path, 1))
    adaptive_jobs = _adaptive_jobs(jobs)

    assert len(jobs) == 3
    assert len(adaptive_jobs) == 2
    assert {_nodes(job) for job in adaptive_jobs} == {30}
    assert {job.definition.search.searches_per_move for job in adaptive_jobs} == {1, 64}
    assert all(job.definition.search.parallel_searches == 1 for job in adaptive_jobs)
    assert all(job.definition.opening_pair_count == 50 for job in adaptive_jobs)
    pickle.dumps(tuple(process.args for process in context.processes))


@pytest.mark.parametrize(
    ('scores', 'expected_nodes'),
    [
        ((0.70,), (30, 100)),
        ((0.69,), (30, 30)),
        ((0.71, 0.69, 0.31, 0.30), (30, 100, 100, 100, 30)),
    ],
)
def test_adaptive_rung_uses_deadband_and_moves_at_most_one_step_per_boundary(
    tmp_path: Path,
    scores: tuple[float, ...],
    expected_nodes: tuple[int, ...],
) -> None:
    experiment = _adaptive_experiment(tmp_path)
    clock = FakeClock()
    context = FakeProcessContext()
    manager = EvaluationManager(experiment, _checkpoint(tmp_path, 0), clock, context)
    observed_nodes: list[int] = []

    for boundary_index, score in enumerate(scores, start=1):
        clock.now = boundary_index * 20 + 1.0
        jobs = manager.schedule_due_jobs(_checkpoint(tmp_path, boundary_index))
        searched_job = next(job for job in _adaptive_jobs(jobs) if job.definition.definition_id == 'stockfish-searched')
        observed_nodes.append(_nodes(searched_job))
        _complete_boundary(manager, context, jobs, score)

    clock.now = (len(scores) + 1) * 20 + 1.0
    next_jobs = manager.schedule_due_jobs(_checkpoint(tmp_path, len(scores) + 1))
    next_searched_job = next(
        job for job in _adaptive_jobs(next_jobs) if job.definition.definition_id == 'stockfish-searched'
    )
    observed_nodes.append(_nodes(next_searched_job))

    assert tuple(observed_nodes) == expected_nodes


def test_adaptive_rung_recovers_completed_result_from_persisted_job(tmp_path: Path) -> None:
    experiment = _adaptive_experiment(tmp_path)
    first_clock = FakeClock()
    first_context = FakeProcessContext()
    first_manager = EvaluationManager(experiment, _checkpoint(tmp_path, 0), first_clock, first_context)
    first_clock.now = 21.0
    jobs = first_manager.schedule_due_jobs(_checkpoint(tmp_path, 1))
    searched_job = next(job for job in _adaptive_jobs(jobs) if job.definition.definition_id == 'stockfish-searched')
    _write_match_result(searched_job, 0.70)
    searched_process = next(
        process
        for process in first_context.processes
        if EVALUATION_JOB_ADAPTER.validate_json(process.args[1]).job_id == searched_job.job_id
    )
    searched_process.exitcode = 0
    searched_job.candidate.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    searched_job.candidate.manifest_path.write_text('{}', encoding='utf-8')
    searched_job.candidate.inference_model_path.write_bytes(b'model')

    restarted = EvaluationManager(
        experiment,
        _checkpoint(tmp_path, 0),
        FakeClock(),
        FakeProcessContext(),
    )
    restarted.start()
    state = EvaluationManagerState.model_validate_json(
        (tmp_path / 'evaluations' / 'manager-state.json').read_text(encoding='utf-8')
    )
    searched_rung = next(rung for rung in state.adaptive_stockfish_rungs if rung.definition_id == 'stockfish-searched')

    assert searched_rung.selected_nodes == 100
    assert searched_rung.last_completed_boundary_seconds == 20


def test_adaptive_stockfish_logs_selected_and_next_nodes_per_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _adaptive_experiment(tmp_path)
    clock = FakeClock()
    context = FakeProcessContext()
    manager = EvaluationManager(experiment, _checkpoint(tmp_path, 0), clock, context)
    scalar_events: list[tuple[str, float, int]] = []
    monkeypatch.setattr(
        'src.evaluation.manager.log_scalar',
        lambda name, value, step: scalar_events.append((name, value, step)),
    )

    clock.now = 21.0
    jobs = manager.schedule_due_jobs(_checkpoint(tmp_path, 1))
    _complete_boundary(manager, context, jobs, searched_score=0.70, policy_score=0.30)

    named = {(name, step): value for name, value, step in scalar_events}
    assert named[('evaluation_metadata/stockfish-searched/stockfish_nodes', 20)] == 30
    assert named[('evaluation_metadata/stockfish-policy-only/stockfish_nodes', 20)] == 30
    assert named[('evaluation_metadata/stockfish-searched/next_stockfish_nodes', 20)] == 100
    assert named[('evaluation_metadata/stockfish-policy-only/next_stockfish_nodes', 20)] == 30
    assert ('evaluation/ladder_elo_64', 20) in named
    assert ('evaluation/ladder_elo_1', 20) in named
