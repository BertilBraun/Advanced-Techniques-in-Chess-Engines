from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import Field
from src.evaluation.configuration import (
    EvaluationSearchConfiguration,
    PreviousCheckpointEvaluationDefinition,
)
from src.evaluation.contracts import (
    OPENING_SUITE_MANIFEST_ADAPTER,
    AnyOpeningSuiteManifest,
    CheckpointOpponent,
    EvaluationGameResult,
    EvaluationTerminationReason,
    MatchAggregate,
    MatchEvaluationJob,
)
from src.evaluation.match import SearchActionSelector, run_match
from src.experiment.configuration import experiment_configuration_sha256, load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.contract import ChessPosition, ChessStateContract
from src.games.chess.training import ChessImplementation
from src.self_play.configuration import BatchedInferenceParams
from src.self_play.native_search import NativeSelfPlaySearch
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.contracts import read_checkpoint_manifest
from src.training.network import NetworkDefinition
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from src.util.provenance import read_source_revision_if_available

ProbeMode: TypeAlias = Literal['equal-nodes', 'equal-compute', 'throughput-only']

# Roots must match what the match itself batches: with parallel_searches 1 the inference batch is the root count,
# and a paired match has about half its games on each side to move, so 400 games present ~200 roots per search.
THROUGHPUT_POSITION_COUNT = 200
THROUGHPUT_WARMUP_BATCHES = 2
THROUGHPUT_MEASURED_BATCHES = 16


@dataclass(frozen=True)
class Arguments:
    teacher_run_state: Path
    teacher_generation: int
    student_run_state: Path
    student_generation: int
    openings_manifest: Path
    mode: ProbeMode
    searches_per_move: int
    parallel_searches: int
    exploration_constant: float
    opening_pair_count: int
    maximum_game_plies: int
    bootstrap_samples: int
    device_id: int
    random_seed: int
    output: Path
    experiment_config: Path
    pinned_throughput_ratio: float | None


class NetworkIdentity(FrozenModel):
    run_state_path: Path
    generation: int = Field(ge=0)
    inference_model_path: Path
    inference_model_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    network: NetworkDefinition


class ThroughputMeasurement(FrozenModel):
    searches_per_move: int = Field(gt=0)
    parallel_searches: int = Field(gt=0)
    position_count: int = Field(gt=0)
    warmup_batches: int = Field(ge=0)
    measured_batches: int = Field(gt=0)
    elapsed_seconds: float = Field(gt=0.0)
    inference_positions: int = Field(ge=0)
    inference_calls: int = Field(ge=0)
    inference_seconds: float = Field(ge=0.0)
    average_positions_per_inference_call: float = Field(ge=0.0)
    worker_utilization: float = Field(ge=0.0)
    positions_per_second: float = Field(ge=0.0)


class SearchBudgets(FrozenModel):
    teacher_searches_per_move: int = Field(gt=0)
    student_searches_per_move: int = Field(gt=0)
    parallel_searches: int = Field(gt=0)
    exploration_constant: float = Field(gt=0.0)
    student_budget_clamped: bool


class EloDifference(FrozenModel):
    point: float | None
    confidence_low: float | None
    confidence_high: float | None


class DistillationMatchResult(FrozenModel):
    schema_version: Literal[1] = 1
    source_revision: str | None
    source_dirty: bool | None
    tool_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    started_at_utc: datetime
    mode: ProbeMode
    experiment_path: Path
    experiment_configuration_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    opening_manifest_path: Path
    opening_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    opening_manifest_pair_count: int = Field(gt=0)
    opening_pair_count: int = Field(gt=0)
    maximum_game_plies: int = Field(gt=0)
    device_id: int = Field(ge=0)
    random_seed: int = Field(ge=0)
    bootstrap_samples: int = Field(gt=0)
    teacher: NetworkIdentity
    student: NetworkIdentity
    teacher_throughput: ThroughputMeasurement | None
    student_throughput: ThroughputMeasurement | None
    throughput_ratio: float | None
    budgets: SearchBudgets | None
    games: tuple[EvaluationGameResult, ...]
    aggregate: MatchAggregate | None
    maximum_plies_games: int | None
    elo_difference: EloDifference | None
    duration_seconds: float = Field(ge=0.0)


def _network_identity(run_state: Path, checkpoint: CheckpointReference) -> NetworkIdentity:
    manifest = read_checkpoint_manifest(checkpoint.generation, run_state)
    return NetworkIdentity(
        run_state_path=run_state.resolve(),
        generation=checkpoint.generation,
        inference_model_path=checkpoint.inference_model_path.resolve(),
        inference_model_sha256=checkpoint.inference_model_sha256,
        network=manifest.network,
    )


def _search_configuration(
    searches_per_move: int,
    parallel_searches: int,
    exploration_constant: float,
    inference: BatchedInferenceParams,
) -> EvaluationSearchConfiguration:
    return EvaluationSearchConfiguration(
        searches_per_move=searches_per_move,
        parallel_searches=parallel_searches,
        exploration_constant=exploration_constant,
        inference=inference,
    )


def _measurement_positions(
    state: ChessStateContract,
    openings: AnyOpeningSuiteManifest,
    position_count: int,
) -> tuple[ChessPosition, ...]:
    positions: list[ChessPosition] = []
    for opening in openings.openings[:position_count]:
        position = state.initial_position()
        for action_id in opening.action_ids:
            position = state.child_position(position, action_id)
        positions.append(position)
    return tuple(positions)


def _run_search_batch(search: NativeSelfPlaySearch, positions: tuple[ChessPosition, ...]) -> None:
    roots = tuple(search.new_root(position) for position in positions)
    search.search([search.request(root, True) for root in roots])


def _measure_throughput(
    game: ChessImplementation,
    checkpoint: CheckpointReference,
    device_id: int,
    configuration: EvaluationSearchConfiguration,
    positions: tuple[ChessPosition, ...],
) -> ThroughputMeasurement:
    search = game.create_evaluation_search(device_id, checkpoint, configuration)
    for _ in range(THROUGHPUT_WARMUP_BATCHES):
        _run_search_batch(search, positions)
    initial = search.inference_statistics()
    started_at = time.perf_counter()
    for _ in range(THROUGHPUT_MEASURED_BATCHES):
        _run_search_batch(search, positions)
    elapsed_seconds = time.perf_counter() - started_at
    final = search.inference_statistics()
    inference_positions = final.modelInferencePositions - initial.modelInferencePositions
    inference_calls = final.modelInferenceCalls - initial.modelInferenceCalls
    inference_nanoseconds = final.inferenceNanoseconds - initial.inferenceNanoseconds
    assert inference_positions >= 0 and inference_calls >= 0 and inference_nanoseconds >= 0
    return ThroughputMeasurement(
        searches_per_move=configuration.searches_per_move,
        parallel_searches=configuration.parallel_searches,
        position_count=len(positions),
        warmup_batches=THROUGHPUT_WARMUP_BATCHES,
        measured_batches=THROUGHPUT_MEASURED_BATCHES,
        elapsed_seconds=elapsed_seconds,
        inference_positions=inference_positions,
        inference_calls=inference_calls,
        inference_seconds=inference_nanoseconds / 1e9,
        average_positions_per_inference_call=final.averageNumberOfPositionsInInferenceCall,
        worker_utilization=final.workerUtilization,
        positions_per_second=inference_positions / elapsed_seconds,
    )


def _equal_compute_student_searches(
    teacher_searches_per_move: int,
    throughput_ratio: float,
    parallel_searches: int,
) -> tuple[int, bool]:
    if throughput_ratio <= 0.0:
        raise ValueError('Throughput ratio must be positive.')
    scaled = round(teacher_searches_per_move * throughput_ratio)
    # EvaluationSearchConfiguration rejects a budget that does not exceed parallel_searches.
    clamped = max(scaled, parallel_searches + 1)
    return clamped, clamped != scaled


def _score_to_elo(score: float) -> float | None:
    if score <= 0.0 or score >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def _elo_difference(aggregate: MatchAggregate) -> EloDifference:
    return EloDifference(
        point=_score_to_elo(aggregate.score),
        confidence_low=_score_to_elo(aggregate.score_confidence_low),
        confidence_high=_score_to_elo(aggregate.score_confidence_high),
    )


def _match_job(
    arguments: Arguments,
    teacher: CheckpointReference,
    student: CheckpointReference,
    teacher_search: EvaluationSearchConfiguration,
) -> MatchEvaluationJob:
    # run_match builds the opponent selector from definition.search, so this carries the teacher's budget;
    # the student's asymmetric budget arrives as the candidate_selector override.
    definition = PreviousCheckpointEvaluationDefinition(
        kind='previous_checkpoint',
        definition_id=f'distill-{arguments.mode}-t{teacher.generation}-s{student.generation}',
        boundary_offset=1,
        boundary_parity='every',
        opening_pair_count=arguments.opening_pair_count,
        maximum_game_plies=arguments.maximum_game_plies,
        search=teacher_search,
    )
    return MatchEvaluationJob(
        kind='match',
        job_id=f'distill-match-{arguments.mode}-g{student.generation}',
        definition=definition,
        boundary_seconds=1,
        candidate=student,
        opponent=CheckpointOpponent(kind='checkpoint', checkpoint=teacher),
        device_id=arguments.device_id,
        deadline_seconds=7 * 24 * 60 * 60,
        random_seed=arguments.random_seed,
        result_path=arguments.output.resolve(),
    )


def _report_throughput(
    teacher_throughput: ThroughputMeasurement,
    student_throughput: ThroughputMeasurement,
    ratio: float,
) -> None:
    print(f'teacher throughput: {teacher_throughput.positions_per_second:.1f} positions/s')
    print(f'student throughput: {student_throughput.positions_per_second:.1f} positions/s')
    print(f'student/teacher throughput ratio: {ratio:.4f}')


def _report_budgets(budgets: SearchBudgets) -> None:
    print(f'teacher searches_per_move: {budgets.teacher_searches_per_move}')
    print(f'student searches_per_move: {budgets.student_searches_per_move}')
    if budgets.student_budget_clamped:
        print(f'student budget clamped up to exceed parallel_searches={budgets.parallel_searches}')


def _report_match(result: DistillationMatchResult) -> None:
    assert result.aggregate is not None and result.elo_difference is not None and result.budgets is not None
    aggregate = result.aggregate
    elo = result.elo_difference
    print(
        f'search budgets: teacher {result.budgets.teacher_searches_per_move}, '
        f'student {result.budgets.student_searches_per_move} searches per move'
    )
    print(f'student W/D/L: {aggregate.wins}/{aggregate.draws}/{aggregate.losses} over {len(result.games)} games')
    print(
        f'student score: {aggregate.score:.4f} '
        f'[{aggregate.score_confidence_low:.4f}, {aggregate.score_confidence_high:.4f}] (95%)'
    )
    print(
        f'student Elo difference: {_format_elo(elo.point)} [{_format_elo(elo.confidence_low)}, '
        f'{_format_elo(elo.confidence_high)}] (95%)'
    )
    print(f'games abandoned at the ply cap: {result.maximum_plies_games}')


def _format_elo(value: float | None) -> str:
    return 'unbounded' if value is None else f'{value:+.1f}'


def _measure_both(
    game: ChessImplementation,
    arguments: Arguments,
    teacher: CheckpointReference,
    student: CheckpointReference,
    openings: AnyOpeningSuiteManifest,
    measurement_search: EvaluationSearchConfiguration,
) -> tuple[ThroughputMeasurement, ThroughputMeasurement]:
    positions = _measurement_positions(
        game.state,
        openings,
        min(THROUGHPUT_POSITION_COUNT, len(openings.openings)),
    )
    teacher_throughput = _measure_throughput(game, teacher, arguments.device_id, measurement_search, positions)
    student_throughput = _measure_throughput(game, student, arguments.device_id, measurement_search, positions)
    return teacher_throughput, student_throughput


def run_probe(arguments: Arguments) -> DistillationMatchResult:
    started_at = time.monotonic()
    started_at_utc = datetime.now(timezone.utc)
    experiment = load_experiment_configuration(arguments.experiment_config)
    if not isinstance(experiment, ChessExperimentConfiguration):
        raise ValueError('Distillation match requires a chess experiment configuration.')
    openings = OPENING_SUITE_MANIFEST_ADAPTER.validate_json(arguments.openings_manifest.read_text(encoding='utf-8'))
    if openings.game != 'chess':
        raise ValueError('Distillation match requires a chess opening manifest.')
    if len(openings.openings) < arguments.opening_pair_count:
        raise ValueError('Opening manifest does not contain the requested number of opening pairs.')
    teacher = CheckpointReference.load_for_inference(arguments.teacher_run_state, arguments.teacher_generation)
    student = CheckpointReference.load_for_inference(arguments.student_run_state, arguments.student_generation)
    game = ChessImplementation(experiment)
    measurement_search = _search_configuration(
        arguments.searches_per_move,
        arguments.parallel_searches,
        arguments.exploration_constant,
        game.self_play_configuration.inference,
    )

    teacher_throughput: ThroughputMeasurement | None = None
    student_throughput: ThroughputMeasurement | None = None
    throughput_ratio: float | None = None
    student_searches_per_move = arguments.searches_per_move
    student_budget_clamped = False
    match arguments.mode:
        case 'equal-nodes':
            pass
        case 'equal-compute' if arguments.pinned_throughput_ratio is not None:
            # Re-measuring per match drew ratios spanning 3.87-5.69 for one model, moving Elo by tens of points.
            throughput_ratio = arguments.pinned_throughput_ratio
            print(f'pinned student/teacher throughput ratio: {throughput_ratio:.4f}')
            student_searches_per_move, student_budget_clamped = _equal_compute_student_searches(
                arguments.searches_per_move,
                throughput_ratio,
                arguments.parallel_searches,
            )
        case 'equal-compute' | 'throughput-only':
            teacher_throughput, student_throughput = _measure_both(
                game,
                arguments,
                teacher,
                student,
                openings,
                measurement_search,
            )
            throughput_ratio = student_throughput.positions_per_second / teacher_throughput.positions_per_second
            _report_throughput(teacher_throughput, student_throughput, throughput_ratio)
            if arguments.mode == 'equal-compute':
                student_searches_per_move, student_budget_clamped = _equal_compute_student_searches(
                    arguments.searches_per_move,
                    throughput_ratio,
                    arguments.parallel_searches,
                )

    budgets: SearchBudgets | None = None
    games: tuple[EvaluationGameResult, ...] = ()
    aggregate: MatchAggregate | None = None
    maximum_plies_games: int | None = None
    elo: EloDifference | None = None
    if arguments.mode != 'throughput-only':
        budgets = SearchBudgets(
            teacher_searches_per_move=arguments.searches_per_move,
            student_searches_per_move=student_searches_per_move,
            parallel_searches=arguments.parallel_searches,
            exploration_constant=arguments.exploration_constant,
            student_budget_clamped=student_budget_clamped,
        )
        _report_budgets(budgets)
        student_search = _search_configuration(
            student_searches_per_move,
            arguments.parallel_searches,
            arguments.exploration_constant,
            game.self_play_configuration.inference,
        )
        candidate_selector: SearchActionSelector[ChessPosition] = SearchActionSelector(
            game.create_evaluation_search(arguments.device_id, student, student_search)
        )
        match_result = run_match(
            _match_job(arguments, teacher, student, measurement_search),
            game,
            openings,
            arguments.bootstrap_samples,
            None,
            experiment.training.topology.trainer.device_type,
            candidate_selector=candidate_selector,
        )
        games = match_result.games
        aggregate = match_result.aggregate
        maximum_plies_games = sum(
            game_result.termination_reason is EvaluationTerminationReason.MAXIMUM_PLIES for game_result in games
        )
        elo = _elo_difference(aggregate)

    revision = read_source_revision_if_available()
    result = DistillationMatchResult(
        source_revision=None if revision is None else revision.commit,
        source_dirty=None if revision is None else revision.dirty,
        tool_sha256=file_sha256(Path(__file__)),
        started_at_utc=started_at_utc,
        mode=arguments.mode,
        experiment_path=arguments.experiment_config.resolve(),
        experiment_configuration_sha256=experiment_configuration_sha256(experiment),
        opening_manifest_path=arguments.openings_manifest.resolve(),
        opening_manifest_sha256=file_sha256(arguments.openings_manifest),
        opening_manifest_pair_count=len(openings.openings),
        opening_pair_count=arguments.opening_pair_count,
        maximum_game_plies=arguments.maximum_game_plies,
        device_id=arguments.device_id,
        random_seed=arguments.random_seed,
        bootstrap_samples=arguments.bootstrap_samples,
        teacher=_network_identity(arguments.teacher_run_state, teacher),
        student=_network_identity(arguments.student_run_state, student),
        teacher_throughput=teacher_throughput,
        student_throughput=student_throughput,
        throughput_ratio=throughput_ratio,
        budgets=budgets,
        games=games,
        aggregate=aggregate,
        maximum_plies_games=maximum_plies_games,
        elo_difference=elo,
        duration_seconds=time.monotonic() - started_at,
    )
    write_text_atomically(arguments.output, result.model_dump_json(indent=2) + '\n')
    return result


def _probe_mode(value: str) -> ProbeMode:
    match value:
        case 'equal-nodes' | 'equal-compute' | 'throughput-only':
            return value
        case _:
            raise ValueError(f'Unknown distillation probe mode: {value}')


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Play a distilled student against its teacher on one GPU.')
    parser.add_argument('--teacher-run-state', required=True, type=Path)
    parser.add_argument('--teacher-generation', required=True, type=int)
    parser.add_argument('--student-run-state', required=True, type=Path)
    parser.add_argument('--student-generation', required=True, type=int)
    parser.add_argument('--openings-manifest', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('equal-nodes', 'equal-compute', 'throughput-only'))
    parser.add_argument('--searches-per-move', default=64, type=int)
    parser.add_argument('--parallel-searches', default=1, type=int)
    parser.add_argument('--exploration-constant', default=1.0, type=float)
    parser.add_argument('--opening-pair-count', default=200, type=int)
    parser.add_argument('--maximum-game-plies', default=300, type=int)
    parser.add_argument('--bootstrap-samples', default=10000, type=int)
    parser.add_argument('--device-id', default=0, type=int)
    parser.add_argument('--random-seed', default=0, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--experiment-config', required=True, type=Path)
    parser.add_argument('--pinned-throughput-ratio', default=None, type=float)
    namespace = parser.parse_args()
    arguments = Arguments(
        teacher_run_state=namespace.teacher_run_state,
        teacher_generation=namespace.teacher_generation,
        student_run_state=namespace.student_run_state,
        student_generation=namespace.student_generation,
        openings_manifest=namespace.openings_manifest,
        mode=_probe_mode(namespace.mode),
        searches_per_move=namespace.searches_per_move,
        parallel_searches=namespace.parallel_searches,
        exploration_constant=namespace.exploration_constant,
        opening_pair_count=namespace.opening_pair_count,
        maximum_game_plies=namespace.maximum_game_plies,
        bootstrap_samples=namespace.bootstrap_samples,
        device_id=namespace.device_id,
        random_seed=namespace.random_seed,
        output=namespace.output,
        experiment_config=namespace.experiment_config,
        pinned_throughput_ratio=namespace.pinned_throughput_ratio,
    )
    required_paths = (
        arguments.teacher_run_state,
        arguments.student_run_state,
        arguments.openings_manifest,
        arguments.experiment_config,
    )
    if not all(path.exists() for path in required_paths):
        raise ValueError('Both run states, the opening manifest, and the experiment configuration must exist.')
    if arguments.teacher_generation < 0 or arguments.student_generation < 0:
        raise ValueError('Checkpoint generations must be nonnegative.')
    if arguments.device_id < 0 or arguments.random_seed < 0:
        raise ValueError('Device ID and random seed must be nonnegative.')
    if arguments.searches_per_move <= arguments.parallel_searches or arguments.parallel_searches <= 0:
        raise ValueError('Searches per move must exceed a positive parallel search count.')
    positive_values = (
        arguments.opening_pair_count,
        arguments.maximum_game_plies,
        arguments.bootstrap_samples,
    )
    if any(value <= 0 for value in positive_values) or arguments.exploration_constant <= 0.0:
        raise ValueError('Opening pairs, ply cap, bootstrap samples, and exploration constant must be positive.')
    if arguments.output.exists():
        raise ValueError(f'Distillation match output already exists: {arguments.output}')
    return arguments


def main() -> None:
    arguments = parse_arguments()
    result = run_probe(arguments)
    if result.aggregate is not None:
        _report_match(result)
    print(f'result written to {arguments.output}')


if __name__ == '__main__':
    main()
