from __future__ import annotations

import argparse
import csv
import random
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field
from src.evaluation.configuration import (
    EvaluationSearchConfiguration,
    StockfishEngineConfiguration,
    StockfishFixedNodesEvaluationDefinition,
)
from src.evaluation.contracts import (
    OPENING_SUITE_MANIFEST_ADAPTER,
    AnyOpeningSuiteManifest,
    EvaluationGameResult,
    MatchAggregate,
    MatchEvaluationJob,
    MatchEvaluationResult,
    StockfishFixedNodesOpponent,
)
from src.evaluation.inference import PolicyActionSelector
from src.evaluation.match import (
    ConcurrentMatchGroup,
    MatchActionSelector,
    SearchActionSelector,
    run_concurrent_matches,
    run_match,
)
from src.evaluation.statistics import aggregate_match
from src.experiment.configuration import experiment_configuration_sha256, load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.contract import ChessPosition
from src.games.chess.interactive.analysis import TimedMctsAnalysis
from src.games.chess.interactive.configuration import InferenceTarget, InteractiveEngineConfiguration
from src.games.chess.interactive.engine import InteractiveEngine
from src.games.chess.stockfish import StockfishClient, StockfishFixedNodesMatchEngine
from src.games.chess.training import ChessImplementation
from src.self_play.configuration import BatchedInferenceParams
from src.training.checkpoint import CheckpointReference
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from src.util.provenance import read_source_revision
from tools.search_budget import (
    DirectPolicyModelBudget,
    FixedModelSearchBudget,
    ModelSearchBudget,
    TimedModelSearchBudget,
    add_model_search_budget_arguments,
    model_search_budget,
)


class PrefixOpeningSelection(FrozenModel):
    kind: Literal['prefix'] = 'prefix'


class SeededOpeningSelection(FrozenModel):
    kind: Literal['seeded_sample'] = 'seeded_sample'
    random_seed: int = Field(ge=0)


OpeningSelection: TypeAlias = Annotated[
    PrefixOpeningSelection | SeededOpeningSelection,
    Field(discriminator='kind'),
]


class GpuProvenance(FrozenModel):
    device_id: int = Field(ge=0)
    uuid: str = Field(min_length=1)
    name: str = Field(min_length=1)
    memory_total_mib: int = Field(gt=0)
    driver_version: str = Field(min_length=1)


class BusyGpuProcess(FrozenModel):
    device_id: int = Field(ge=0)
    gpu_uuid: str = Field(min_length=1)
    process_id: int = Field(gt=0)
    process_name: str = Field(min_length=1)


class TimedMoveMeasurements(FrozenModel):
    move_count: int = Field(ge=0)
    total_searches: int = Field(ge=0)
    minimum_searches: int = Field(ge=0)
    maximum_searches: int = Field(ge=0)
    mean_searches: float = Field(ge=0.0)
    total_elapsed_milliseconds: int = Field(ge=0)
    minimum_elapsed_milliseconds: int = Field(ge=0)
    maximum_elapsed_milliseconds: int = Field(ge=0)
    mean_elapsed_milliseconds: float = Field(ge=0.0)


class GauntletShardResult(FrozenModel):
    shard_id: int = Field(ge=0)
    device_id: int = Field(ge=0)
    first_pair_index: int = Field(ge=0)
    pair_count: int = Field(gt=0)
    stockfish_identity: str = Field(min_length=1)
    games: tuple[EvaluationGameResult, ...]
    timed_move_measurements: TimedMoveMeasurements | None
    duration_seconds: float = Field(ge=0.0)


class StockfishGauntletResult(FrozenModel):
    schema_version: Literal[3] = 3
    source_revision: str = Field(min_length=40, max_length=40)
    tool_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    started_at_utc: datetime
    experiment_path: Path
    experiment_configuration_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    run_directory: Path
    evaluated_checkpoint: CheckpointReference
    opening_manifest_path: Path
    opening_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    opening_manifest_pair_count: int = Field(gt=0)
    opening_pair_count: int = Field(gt=0)
    opening_selection: OpeningSelection
    selected_opening_indices: tuple[int, ...] = Field(min_length=1)
    match_random_seed: int = Field(ge=0)
    stockfish_executable_path: Path
    stockfish_executable_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    stockfish_identity: str = Field(min_length=1)
    stockfish_match_nodes: int = Field(gt=0)
    stockfish_threads: int = Field(gt=0)
    stockfish_hash_mib: int = Field(gt=0)
    model_search_budget: ModelSearchBudget
    gpus: tuple[GpuProvenance, ...] = Field(min_length=1)
    idle_device_check_enforced: bool
    timed_move_measurements: TimedMoveMeasurements | None
    games: tuple[EvaluationGameResult, ...]
    aggregate: MatchAggregate
    shards: tuple[GauntletShardResult, ...]
    duration_seconds: float = Field(ge=0.0)


@dataclass(frozen=True)
class Arguments:
    experiment: Path
    run_directory: Path
    checkpoint_generation: int
    opening_manifest: Path
    stockfish_executable: Path
    stockfish_nodes: int
    opening_pairs: int
    opening_selection: PrefixOpeningSelection | SeededOpeningSelection
    match_random_seed: int | None
    devices: tuple[int, ...]
    model_search_budget: DirectPolicyModelBudget | FixedModelSearchBudget | TimedModelSearchBudget
    output_directory: Path


@dataclass(frozen=True)
class GauntletRung:
    stockfish_nodes: int
    output_directory: Path


@dataclass(frozen=True)
class ConcurrentGauntletArguments:
    experiment: Path
    run_directory: Path
    checkpoint_generation: int
    opening_manifest: Path
    stockfish_executable: Path
    rungs: tuple[GauntletRung, ...]
    opening_pairs: int
    opening_selection: PrefixOpeningSelection | SeededOpeningSelection
    match_random_seed: int | None
    devices: tuple[int, ...]
    model_search_budget: DirectPolicyModelBudget | FixedModelSearchBudget | TimedModelSearchBudget


@dataclass(frozen=True)
class _ShardRung:
    stockfish_nodes: int
    match_random_seed: int
    output_path: Path


@dataclass(frozen=True)
class _ShardRequest:
    shard_id: int
    device_id: int
    first_pair_index: int
    pair_count: int
    experiment: Path
    run_directory: Path
    checkpoint_generation: int
    opening_manifest: Path
    stockfish_executable: Path
    rungs: tuple[_ShardRung, ...]
    opening_indices: tuple[int, ...]
    model_search_budget: DirectPolicyModelBudget | FixedModelSearchBudget | TimedModelSearchBudget


class _TimedSearchActionSelector(MatchActionSelector[ChessPosition]):
    def __init__(
        self,
        checkpoint: CheckpointReference,
        device_id: int,
        budget: TimedModelSearchBudget,
    ) -> None:
        self._seconds_per_move = budget.seconds_per_move
        self._engine = InteractiveEngine(
            InteractiveEngineConfiguration(
                model_path=str(checkpoint.inference_model_path),
                device_id=device_id,
                parallel_searches=budget.parallel_searches,
                exploration_constant=budget.exploration_constant,
                inference_workers=budget.inference_workers,
                outstanding_batches_per_worker=budget.outstanding_batches_per_worker,
                maximum_batch_size=budget.inference_batch_size,
                inference_target=InferenceTarget.CUDA,
            )
        )
        self._searches: list[int] = []
        self._elapsed_milliseconds: list[int] = []

    def choose_actions(self, positions: tuple[ChessPosition, ...]) -> tuple[int, ...]:
        selected: list[int] = []
        for position in positions:
            result = self._engine.new_game(position.fen, ()).analyze(TimedMctsAnalysis(seconds=self._seconds_per_move))
            selected.append(position.action_id_from_uci(result.chosen_move_uci))
            self._searches.append(result.searches)
            self._elapsed_milliseconds.append(result.elapsed_milliseconds)
        return tuple(selected)

    def measurements(self) -> TimedMoveMeasurements:
        if not self._searches:
            return TimedMoveMeasurements(
                move_count=0,
                total_searches=0,
                minimum_searches=0,
                maximum_searches=0,
                mean_searches=0.0,
                total_elapsed_milliseconds=0,
                minimum_elapsed_milliseconds=0,
                maximum_elapsed_milliseconds=0,
                mean_elapsed_milliseconds=0.0,
            )
        return TimedMoveMeasurements(
            move_count=len(self._searches),
            total_searches=sum(self._searches),
            minimum_searches=min(self._searches),
            maximum_searches=max(self._searches),
            mean_searches=sum(self._searches) / len(self._searches),
            total_elapsed_milliseconds=sum(self._elapsed_milliseconds),
            minimum_elapsed_milliseconds=min(self._elapsed_milliseconds),
            maximum_elapsed_milliseconds=max(self._elapsed_milliseconds),
            mean_elapsed_milliseconds=sum(self._elapsed_milliseconds) / len(self._elapsed_milliseconds),
        )


def _run_nvidia_smi(query: str) -> tuple[tuple[str, ...], ...]:
    completed = subprocess.run(
        ('nvidia-smi', f'--query-{query}', '--format=csv,noheader,nounits'),
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(tuple(value.strip() for value in row) for row in csv.reader(completed.stdout.splitlines()) if row)


def _gpu_inventory(devices: tuple[int, ...]) -> tuple[GpuProvenance, ...]:
    records = {
        int(index): GpuProvenance(
            device_id=int(index),
            uuid=uuid,
            name=name,
            memory_total_mib=int(memory_total_mib),
            driver_version=driver_version,
        )
        for index, uuid, name, memory_total_mib, driver_version in _run_nvidia_smi(
            'gpu=index,uuid,name,memory.total,driver_version'
        )
    }
    missing = tuple(device for device in devices if device not in records)
    if missing:
        raise ValueError(f'Requested CUDA devices do not exist: {missing}')
    return tuple(records[device] for device in devices)


def _busy_gpu_processes(gpus: tuple[GpuProvenance, ...]) -> tuple[BusyGpuProcess, ...]:
    device_by_uuid = {gpu.uuid: gpu.device_id for gpu in gpus}
    processes: list[BusyGpuProcess] = []
    for row in _run_nvidia_smi('compute-apps=gpu_uuid,pid,process_name'):
        if len(row) != 3 or row[0] not in device_by_uuid:
            continue
        processes.append(
            BusyGpuProcess(
                device_id=device_by_uuid[row[0]],
                gpu_uuid=row[0],
                process_id=int(row[1]),
                process_name=row[2],
            )
        )
    return tuple(processes)


def _search_configuration(
    budget: DirectPolicyModelBudget | FixedModelSearchBudget | TimedModelSearchBudget,
) -> EvaluationSearchConfiguration:
    # Timed budgets ignore this count, but EvaluationSearchConfiguration demands it exceed parallel_searches.
    match budget:
        case FixedModelSearchBudget():
            searches_per_move = budget.searches_per_move
            parallel_searches = budget.parallel_searches
            exploration_constant = budget.exploration_constant
        case TimedModelSearchBudget():
            searches_per_move = budget.parallel_searches + 1
            parallel_searches = budget.parallel_searches
            exploration_constant = budget.exploration_constant
        case DirectPolicyModelBudget():
            searches_per_move = 2
            parallel_searches = 1
            exploration_constant = 1.0
    return EvaluationSearchConfiguration(
        searches_per_move=searches_per_move,
        parallel_searches=parallel_searches,
        exploration_constant=exploration_constant,
        inference=BatchedInferenceParams(
            inference_workers=budget.inference_workers,
            inference_batch_size=budget.inference_batch_size,
            outstanding_batches_per_worker=budget.outstanding_batches_per_worker,
        ),
    )


def _stockfish_configuration(
    configuration: ChessExperimentConfiguration,
    executable: Path,
    match_nodes: int,
) -> StockfishEngineConfiguration:
    engine = configuration.evaluation.engine
    if not isinstance(engine, StockfishEngineConfiguration):
        raise ValueError('Chess gauntlet requires Stockfish engine configuration.')
    return StockfishEngineConfiguration(
        kind='stockfish',
        executable_path=str(executable.resolve()),
        label_nodes=engine.label_nodes,
        match_nodes=match_nodes,
        threads=engine.threads,
        hash_mib=engine.hash_mib,
        multi_pv=engine.multi_pv,
        policy_softmax_temperature=engine.policy_softmax_temperature,
    )


def _pair_shards(opening_pairs: int, devices: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
    shard_count = min(opening_pairs, len(devices))
    quotient, remainder = divmod(opening_pairs, shard_count)
    first_pair_index = 0
    shards: list[tuple[int, int, int]] = []
    for shard_id, device_id in enumerate(devices[:shard_count]):
        pair_count = quotient + (1 if shard_id < remainder else 0)
        shards.append((device_id, first_pair_index, pair_count))
        first_pair_index += pair_count
    assert first_pair_index == opening_pairs
    return tuple(shards)


def _select_opening_indices(
    manifest_pair_count: int,
    opening_pairs: int,
    selection: PrefixOpeningSelection | SeededOpeningSelection,
) -> tuple[int, ...]:
    if opening_pairs > manifest_pair_count:
        raise ValueError('Opening manifest does not contain the requested number of opening pairs.')
    match selection:
        case PrefixOpeningSelection():
            return tuple(range(opening_pairs))
        case SeededOpeningSelection(random_seed=random_seed):
            generator = random.Random(random_seed)
            return tuple(sorted(generator.sample(range(manifest_pair_count), opening_pairs)))


def _shift_game_indices(game: EvaluationGameResult, first_pair_index: int) -> EvaluationGameResult:
    return game.model_copy(
        update={
            'game_index': game.game_index + 2 * first_pair_index,
            'pair_index': game.pair_index + first_pair_index,
        }
    )


def _combine_timed_measurements(shards: tuple[GauntletShardResult, ...]) -> TimedMoveMeasurements | None:
    measurements = tuple(
        shard.timed_move_measurements
        for shard in shards
        if shard.timed_move_measurements is not None and shard.timed_move_measurements.move_count > 0
    )
    if not measurements:
        return None
    move_count = sum(measurement.move_count for measurement in measurements)
    total_searches = sum(measurement.total_searches for measurement in measurements)
    total_elapsed_milliseconds = sum(measurement.total_elapsed_milliseconds for measurement in measurements)
    return TimedMoveMeasurements(
        move_count=move_count,
        total_searches=total_searches,
        minimum_searches=min(measurement.minimum_searches for measurement in measurements),
        maximum_searches=max(measurement.maximum_searches for measurement in measurements),
        mean_searches=total_searches / move_count,
        total_elapsed_milliseconds=total_elapsed_milliseconds,
        minimum_elapsed_milliseconds=min(measurement.minimum_elapsed_milliseconds for measurement in measurements),
        maximum_elapsed_milliseconds=max(measurement.maximum_elapsed_milliseconds for measurement in measurements),
        mean_elapsed_milliseconds=total_elapsed_milliseconds / move_count,
    )


@dataclass(frozen=True)
class _ShardContext:
    configuration: ChessExperimentConfiguration
    checkpoint: CheckpointReference
    game: ChessImplementation
    openings: AnyOpeningSuiteManifest
    search: EvaluationSearchConfiguration


def _shard_context(request: _ShardRequest) -> _ShardContext:
    loaded = load_experiment_configuration(request.experiment)
    if not isinstance(loaded, ChessExperimentConfiguration):
        raise ValueError('Stockfish gauntlet requires a chess experiment.')
    openings = OPENING_SUITE_MANIFEST_ADAPTER.validate_json(request.opening_manifest.read_text(encoding='utf-8'))
    if openings.game != 'chess':
        raise ValueError('Stockfish gauntlet requires chess openings.')
    return _ShardContext(
        configuration=loaded,
        checkpoint=CheckpointReference.load_for_inference(request.run_directory, request.checkpoint_generation),
        game=ChessImplementation(loaded),
        openings=openings.model_copy(
            update={'openings': tuple(openings.openings[index] for index in request.opening_indices)}
        ),
        search=_search_configuration(request.model_search_budget),
    )


def _shard_job(request: _ShardRequest, context: _ShardContext, rung: _ShardRung) -> MatchEvaluationJob:
    return MatchEvaluationJob(
        kind='match',
        job_id=f'stockfish-n{rung.stockfish_nodes}-g{context.checkpoint.generation}-shard{request.shard_id}',
        definition=StockfishFixedNodesEvaluationDefinition(
            kind='stockfish_fixed_nodes',
            definition_id=f'stockfish-fixed-nodes-{rung.stockfish_nodes}',
            nodes=rung.stockfish_nodes,
            opening_pair_count=request.pair_count,
            maximum_game_plies=300,
            search=context.search,
        ),
        boundary_seconds=1,
        candidate=context.checkpoint,
        opponent=StockfishFixedNodesOpponent(kind='stockfish_fixed_nodes', nodes=rung.stockfish_nodes),
        device_id=request.device_id,
        deadline_seconds=7 * 24 * 60 * 60,
        random_seed=rung.match_random_seed + request.first_pair_index,
        result_path=rung.output_path,
    )


def _open_stockfish_engine(
    request: _ShardRequest,
    context: _ShardContext,
    rung: _ShardRung,
) -> StockfishFixedNodesMatchEngine:
    engine_configuration = _stockfish_configuration(
        context.configuration,
        request.stockfish_executable,
        rung.stockfish_nodes,
    )
    client = StockfishClient(engine_configuration, context.game.state, request.stockfish_executable.resolve())
    return StockfishFixedNodesMatchEngine(client, rung.stockfish_nodes)


def _shard_result(
    request: _ShardRequest,
    match: MatchEvaluationResult,
    stockfish_identity: str,
    timed_measurements: TimedMoveMeasurements | None,
    duration_seconds: float,
) -> GauntletShardResult:
    result = GauntletShardResult(
        shard_id=request.shard_id,
        device_id=request.device_id,
        first_pair_index=request.first_pair_index,
        pair_count=request.pair_count,
        stockfish_identity=stockfish_identity,
        games=tuple(_shift_game_indices(game_result, request.first_pair_index) for game_result in match.games),
        timed_move_measurements=timed_measurements,
        duration_seconds=duration_seconds,
    )
    write_text_atomically(match.job.result_path, result.model_dump_json(indent=2) + '\n')
    return result


def _run_timed_shard_rungs(request: _ShardRequest, context: _ShardContext) -> tuple[GauntletShardResult, ...]:
    # Timed budgets analyse one position at a time, so concurrency across rungs buys no batching and would
    # only blur the per-rung move measurements. They stay sequential, exactly as the single-rung gauntlet ran.
    assert isinstance(request.model_search_budget, TimedModelSearchBudget)
    results: list[GauntletShardResult] = []
    for rung in request.rungs:
        rung_started_at = time.monotonic()
        selector = _TimedSearchActionSelector(context.checkpoint, request.device_id, request.model_search_budget)
        engine = _open_stockfish_engine(request, context, rung)
        stockfish_identity = engine.client.engine_identity
        try:
            match = run_match(
                _shard_job(request, context, rung),
                context.game,
                context.openings,
                1,
                engine,
                context.configuration.training.topology.trainer.device_type,
                candidate_selector=selector,
            )
        finally:
            engine.close()
        results.append(
            _shard_result(
                request,
                match,
                stockfish_identity,
                selector.measurements(),
                time.monotonic() - rung_started_at,
            )
        )
    return tuple(results)


def _run_concurrent_shard_rungs(request: _ShardRequest, context: _ShardContext) -> tuple[GauntletShardResult, ...]:
    setup_started_at = time.monotonic()
    candidate_selector: MatchActionSelector[ChessPosition] | None = None
    if isinstance(request.model_search_budget, DirectPolicyModelBudget):
        candidate_selector = PolicyActionSelector(
            context.game.state,
            context.checkpoint.inference_model_path,
            request.device_id,
            context.configuration.training.topology.trainer.device_type,
            maximum_batch_size=request.model_search_budget.inference_batch_size,
        )
    elif request.model_search_budget.tree_search is not None:
        candidate_selector = SearchActionSelector(
            context.game.create_evaluation_search(
                request.device_id,
                context.checkpoint,
                context.search,
                request.model_search_budget.tree_search,
            ),
            context.search.searches_per_move,
            context.search.parallel_searches,
        )
    engines = tuple(_open_stockfish_engine(request, context, rung) for rung in request.rungs)
    identities = tuple(engine.client.engine_identity for engine in engines)
    setup_seconds = time.monotonic() - setup_started_at
    try:
        matches = run_concurrent_matches(
            tuple(
                ConcurrentMatchGroup(
                    job=_shard_job(request, context, rung),
                    openings=context.openings,
                    external_engine=engine,
                )
                for rung, engine in zip(request.rungs, engines, strict=True)
            ),
            context.game,
            1,
            context.configuration.training.topology.trainer.device_type,
            candidate_selector=candidate_selector,
        )
    finally:
        for engine in engines:
            engine.close()
    return tuple(
        _shard_result(request, match, identity, None, setup_seconds + match.duration_seconds)
        for match, identity in zip(matches, identities, strict=True)
    )


def _run_shard(request: _ShardRequest) -> tuple[GauntletShardResult, ...]:
    context = _shard_context(request)
    if isinstance(request.model_search_budget, TimedModelSearchBudget):
        return _run_timed_shard_rungs(request, context)
    return _run_concurrent_shard_rungs(request, context)


def _rung_random_seed(
    arguments: ConcurrentGauntletArguments,
    default_random_seed: int,
    stockfish_nodes: int,
) -> int:
    if arguments.match_random_seed is None:
        return default_random_seed + stockfish_nodes
    return arguments.match_random_seed


def _rung_shard_requests(
    arguments: ConcurrentGauntletArguments,
    default_random_seed: int,
    selected_opening_indices: tuple[int, ...],
) -> tuple[_ShardRequest, ...]:
    pair_shards = _pair_shards(arguments.opening_pairs, arguments.devices)
    shard_rungs: list[list[_ShardRung]] = [[] for _ in pair_shards]
    for rung in arguments.rungs:
        rung.output_directory.mkdir(parents=True, exist_ok=False)
        shard_directory = rung.output_directory / 'shards'
        shard_directory.mkdir()
        for shard_id in range(len(pair_shards)):
            shard_rungs[shard_id].append(
                _ShardRung(
                    stockfish_nodes=rung.stockfish_nodes,
                    match_random_seed=_rung_random_seed(arguments, default_random_seed, rung.stockfish_nodes),
                    output_path=shard_directory / f'shard-{shard_id:02d}.json',
                )
            )
    return tuple(
        _ShardRequest(
            shard_id=shard_id,
            device_id=device_id,
            first_pair_index=first_pair_index,
            pair_count=pair_count,
            experiment=arguments.experiment.resolve(),
            run_directory=arguments.run_directory.resolve(),
            checkpoint_generation=arguments.checkpoint_generation,
            opening_manifest=arguments.opening_manifest.resolve(),
            stockfish_executable=arguments.stockfish_executable.resolve(),
            rungs=tuple(shard_rungs[shard_id]),
            opening_indices=selected_opening_indices[first_pair_index : first_pair_index + pair_count],
            model_search_budget=arguments.model_search_budget,
        )
        for shard_id, (device_id, first_pair_index, pair_count) in enumerate(pair_shards)
    )


def _rung_result(
    arguments: ConcurrentGauntletArguments,
    rung: GauntletRung,
    configuration: ChessExperimentConfiguration,
    checkpoint: CheckpointReference,
    manifest_pair_count: int,
    selected_opening_indices: tuple[int, ...],
    gpus: tuple[GpuProvenance, ...],
    idle_check_enforced: bool,
    shards: tuple[GauntletShardResult, ...],
    started_at_utc: datetime,
    duration_seconds: float,
) -> StockfishGauntletResult:
    identities = {shard.stockfish_identity for shard in shards}
    if len(identities) != 1:
        raise ValueError(f'Stockfish worker identities disagree: {sorted(identities)}')
    games = tuple(sorted((game for shard in shards for game in shard.games), key=lambda game: game.game_index))
    if tuple(game.game_index for game in games) != tuple(range(2 * arguments.opening_pairs)):
        raise ValueError('Merged gauntlet games do not cover the expected indices exactly once.')
    match_random_seed = _rung_random_seed(arguments, configuration.training.random_seed, rung.stockfish_nodes)
    engine_configuration = _stockfish_configuration(
        configuration,
        arguments.stockfish_executable,
        rung.stockfish_nodes,
    )
    result = StockfishGauntletResult(
        source_revision=read_source_revision().commit,
        tool_sha256=file_sha256(Path(__file__)),
        started_at_utc=started_at_utc,
        experiment_path=arguments.experiment.resolve(),
        experiment_configuration_sha256=experiment_configuration_sha256(configuration),
        run_directory=arguments.run_directory.resolve(),
        evaluated_checkpoint=checkpoint,
        opening_manifest_path=arguments.opening_manifest.resolve(),
        opening_manifest_sha256=file_sha256(arguments.opening_manifest),
        opening_manifest_pair_count=manifest_pair_count,
        opening_pair_count=arguments.opening_pairs,
        opening_selection=arguments.opening_selection,
        selected_opening_indices=selected_opening_indices,
        match_random_seed=match_random_seed,
        stockfish_executable_path=arguments.stockfish_executable.resolve(),
        stockfish_executable_sha256=file_sha256(arguments.stockfish_executable),
        stockfish_identity=next(iter(identities)),
        stockfish_match_nodes=rung.stockfish_nodes,
        stockfish_threads=engine_configuration.threads,
        stockfish_hash_mib=engine_configuration.hash_mib,
        model_search_budget=arguments.model_search_budget,
        gpus=gpus,
        idle_device_check_enforced=idle_check_enforced,
        timed_move_measurements=_combine_timed_measurements(shards),
        games=games,
        aggregate=aggregate_match(games, match_random_seed, configuration.evaluation.bootstrap_samples),
        shards=shards,
        duration_seconds=duration_seconds,
    )
    write_text_atomically(rung.output_directory / 'result.json', result.model_dump_json(indent=2) + '\n')
    return result


def run_gauntlets(arguments: ConcurrentGauntletArguments) -> tuple[StockfishGauntletResult, ...]:
    """Every rung shares one candidate population per shard, so a single inference batch spans all rungs."""
    started_at = time.monotonic()
    started_at_utc = datetime.now(timezone.utc)
    if not arguments.rungs:
        raise ValueError('A Stockfish gauntlet needs at least one node rung.')
    if len({rung.stockfish_nodes for rung in arguments.rungs}) != len(arguments.rungs):
        raise ValueError('Stockfish gauntlet node rungs must be unique.')
    loaded = load_experiment_configuration(arguments.experiment)
    if not isinstance(loaded, ChessExperimentConfiguration):
        raise ValueError('Stockfish gauntlet requires a chess experiment.')
    if loaded.training.topology.trainer.device_type != 'cuda':
        raise ValueError('Multi-GPU Stockfish gauntlet requires a CUDA experiment.')
    checkpoint = CheckpointReference.load_for_inference(
        arguments.run_directory,
        arguments.checkpoint_generation,
    )
    openings = OPENING_SUITE_MANIFEST_ADAPTER.validate_json(arguments.opening_manifest.read_text(encoding='utf-8'))
    if openings.game != 'chess':
        raise ValueError('Opening manifest must contain the requested number of chess opening pairs.')
    selected_opening_indices = _select_opening_indices(
        len(openings.openings),
        arguments.opening_pairs,
        arguments.opening_selection,
    )
    gpus = _gpu_inventory(arguments.devices)
    idle_check_enforced = isinstance(arguments.model_search_budget, TimedModelSearchBudget)
    if idle_check_enforced:
        busy = _busy_gpu_processes(gpus)
        if busy:
            details = ', '.join(f'GPU {process.device_id}: PID {process.process_id}' for process in busy)
            raise ValueError(f'Timed gauntlet requires idle selected GPUs; found {details}.')

    shard_requests = _rung_shard_requests(arguments, loaded.training.random_seed, selected_opening_indices)
    shards_by_rung: dict[int, list[GauntletShardResult]] = {rung.stockfish_nodes: [] for rung in arguments.rungs}
    with ProcessPoolExecutor(
        max_workers=len(shard_requests),
        mp_context=get_context('spawn'),
    ) as executor:
        futures = {executor.submit(_run_shard, request): request for request in shard_requests}
        for future in as_completed(futures):
            for shard_rung, shard in zip(futures[future].rungs, future.result(), strict=True):
                shards_by_rung[shard_rung.stockfish_nodes].append(shard)
    duration_seconds = time.monotonic() - started_at
    return tuple(
        _rung_result(
            arguments,
            rung,
            loaded,
            checkpoint,
            len(openings.openings),
            selected_opening_indices,
            gpus,
            idle_check_enforced,
            tuple(sorted(shards_by_rung[rung.stockfish_nodes], key=lambda shard: shard.shard_id)),
            started_at_utc,
            duration_seconds,
        )
        for rung in arguments.rungs
    )


def run_gauntlet(arguments: Arguments) -> StockfishGauntletResult:
    return run_gauntlets(
        ConcurrentGauntletArguments(
            experiment=arguments.experiment,
            run_directory=arguments.run_directory,
            checkpoint_generation=arguments.checkpoint_generation,
            opening_manifest=arguments.opening_manifest,
            stockfish_executable=arguments.stockfish_executable,
            rungs=(
                GauntletRung(
                    stockfish_nodes=arguments.stockfish_nodes,
                    output_directory=arguments.output_directory,
                ),
            ),
            opening_pairs=arguments.opening_pairs,
            opening_selection=arguments.opening_selection,
            match_random_seed=arguments.match_random_seed,
            devices=arguments.devices,
            model_search_budget=arguments.model_search_budget,
        )
    )[0]


def _opening_selection(namespace: argparse.Namespace) -> PrefixOpeningSelection | SeededOpeningSelection:
    if namespace.opening_selection == 'prefix':
        if namespace.opening_selection_seed is not None:
            raise ValueError('--opening-selection-seed is only valid with seeded_sample selection.')
        return PrefixOpeningSelection()
    if namespace.opening_selection_seed is None:
        raise ValueError('seeded_sample selection requires --opening-selection-seed.')
    return SeededOpeningSelection(random_seed=namespace.opening_selection_seed)


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Run a paired multi-GPU model-versus-Stockfish gauntlet.')
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--run-directory', required=True, type=Path)
    parser.add_argument('--checkpoint-generation', required=True, type=int)
    parser.add_argument('--opening-manifest', required=True, type=Path)
    parser.add_argument('--stockfish-executable', required=True, type=Path)
    parser.add_argument('--stockfish-nodes', required=True, type=int)
    opening_scope = parser.add_mutually_exclusive_group()
    opening_scope.add_argument('--opening-pairs', type=int)
    opening_scope.add_argument('--all-opening-pairs', action='store_true')
    parser.add_argument('--opening-selection', choices=('prefix', 'seeded_sample'), default='prefix')
    parser.add_argument('--opening-selection-seed', type=int)
    parser.add_argument('--match-random-seed', type=int)
    parser.add_argument('--devices', required=True, nargs='+', type=int)
    add_model_search_budget_arguments(parser)
    parser.add_argument('--output-directory', required=True, type=Path)
    namespace = parser.parse_args()
    if namespace.all_opening_pairs:
        opening_manifest = OPENING_SUITE_MANIFEST_ADAPTER.validate_json(
            namespace.opening_manifest.read_text(encoding='utf-8')
        )
        opening_pairs = len(opening_manifest.openings)
    else:
        opening_pairs = 50 if namespace.opening_pairs is None else namespace.opening_pairs
    arguments = Arguments(
        experiment=namespace.experiment,
        run_directory=namespace.run_directory,
        checkpoint_generation=namespace.checkpoint_generation,
        opening_manifest=namespace.opening_manifest,
        stockfish_executable=namespace.stockfish_executable,
        stockfish_nodes=namespace.stockfish_nodes,
        opening_pairs=opening_pairs,
        opening_selection=_opening_selection(namespace),
        match_random_seed=namespace.match_random_seed,
        devices=tuple(namespace.devices),
        model_search_budget=model_search_budget(namespace),
        output_directory=namespace.output_directory,
    )
    required_paths = (
        arguments.experiment,
        arguments.run_directory,
        arguments.opening_manifest,
        arguments.stockfish_executable,
    )
    if not all(path.exists() for path in required_paths):
        raise ValueError('Experiment, run directory, openings, and Stockfish executable must exist.')
    positive_values = (arguments.stockfish_nodes, arguments.opening_pairs)
    if any(value <= 0 for value in positive_values):
        raise ValueError('Stockfish nodes and opening pairs must be positive.')
    if (
        arguments.checkpoint_generation < 0
        or any(device < 0 for device in arguments.devices)
        or (arguments.match_random_seed is not None and arguments.match_random_seed < 0)
    ):
        raise ValueError('Checkpoint generation, match seed, and device IDs must be nonnegative.')
    if not arguments.devices or len(set(arguments.devices)) != len(arguments.devices):
        raise ValueError('Gauntlet devices must be nonempty and unique.')
    if arguments.output_directory.exists():
        raise ValueError(f'Gauntlet output directory already exists: {arguments.output_directory}')
    _search_configuration(arguments.model_search_budget)
    return arguments


def main() -> None:
    print(run_gauntlet(parse_arguments()).model_dump_json(indent=2))


if __name__ == '__main__':
    main()
