from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from src.self_play.value_target import REPLAY_SCHEMA_VERSION
from src.train.RollingReplayBuffer import (
    CompactionStepStatus,
    ReplayQuantumRequest,
    ReplayShardManifest,
    ReplayTrainingQuantum,
    RollingReplayBuffer,
    TerminationCounts,
    file_sha256,
    prefetch_rank_quanta,
)
from tools.production_ddp_fixture import write_replay_fixture


MATURE_REPLAY_POSITIONS = 2_500_000
MEASURED_POSITIONS_PER_GAME = 42.53
PRODUCTION_GAMES_PER_SHARD = 10
PRODUCTION_POSITIONS_PER_SHARD = 500
WORKER_READY_TIMEOUT_SECONDS = 120.0
WORKER_RESULT_TIMEOUT_SECONDS = 1_800.0


@dataclass(frozen=True)
class Arguments:
    workspace: Path
    output: Path
    shard_count: int
    samples_per_shard: int
    global_batch_size: int
    world_size: int
    optimizer_steps: int
    quantum_count: int
    trainer_consumption_samples_per_second: float
    sampler_seed: int


@dataclass(frozen=True)
class ProcessDecodeResult:
    rank: int
    decoded_samples: int
    elapsed_seconds: float
    maximum_resident_quantum_bytes: int
    payload_opens: int
    selected_rows: int
    rows_read: int
    selected_bytes: int
    bytes_read: int


class WorkerReady(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    kind: Literal['ready'] = 'ready'


class WorkerSuccess(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    kind: Literal['success'] = 'success'
    result: ProcessDecodeResult


class WorkerFailure(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    kind: Literal['failure'] = 'failure'
    exception_type: str
    message: str


WorkerMessage = Annotated[WorkerReady | WorkerSuccess | WorkerFailure, Field(discriminator='kind')]
WORKER_MESSAGE_ADAPTER: TypeAdapter[WorkerMessage] = TypeAdapter(WorkerMessage)


@dataclass(frozen=True)
class CoordinatedDecodeBenchmark:
    wall_seconds: float
    global_samples_per_second: float
    decoded_samples: int
    maximum_rank_quantum_bytes: int
    payload_opens: int
    selected_rows: int
    rows_read: int
    row_read_amplification: float
    selected_bytes: int
    bytes_read: int
    byte_read_amplification: float
    process_results: tuple[ProcessDecodeResult, ...]


@dataclass(frozen=True)
class ReplayLoaderBenchmark:
    replay_capacity_unique_positions: int
    actual_unique_positions: int
    producer_shards_before_compaction: int
    compacted_containers: int
    logical_segments_after_compaction: int
    metadata_memory_bytes_before_compaction: int
    metadata_memory_bytes_after_compaction: int
    assumed_games_per_producer_shard: int
    measured_positions_per_game: float
    compaction_seconds: float
    compaction_preserved_sample_count: bool
    global_batch_size: int
    world_size: int
    optimizer_steps: int
    quantum_count: int
    rank_positions_per_quantum: int
    trainer_global_consumption_samples_per_second: float
    uncompacted: CoordinatedDecodeBenchmark
    compacted: CoordinatedDecodeBenchmark
    compacted_throughput_exceeds_consumption: bool


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-count', type=int, default=5_000)
    parser.add_argument('--samples-per-shard', type=int, default=PRODUCTION_POSITIONS_PER_SHARD)
    parser.add_argument('--global-batch-size', type=int, default=1_024)
    parser.add_argument('--world-size', type=int, default=4)
    parser.add_argument('--optimizer-steps', type=int, default=50)
    parser.add_argument('--quantum-count', type=int, default=2)
    parser.add_argument('--trainer-consumption-samples-per-second', type=float, required=True)
    parser.add_argument('--sampler-seed', type=int, default=20260723)
    namespace = parser.parse_args()
    arguments = Arguments(
        workspace=namespace.workspace,
        output=namespace.output,
        shard_count=namespace.shard_count,
        samples_per_shard=namespace.samples_per_shard,
        global_batch_size=namespace.global_batch_size,
        world_size=namespace.world_size,
        optimizer_steps=namespace.optimizer_steps,
        quantum_count=namespace.quantum_count,
        trainer_consumption_samples_per_second=namespace.trainer_consumption_samples_per_second,
        sampler_seed=namespace.sampler_seed,
    )
    if arguments.shard_count <= 0 or arguments.samples_per_shard <= 0:
        raise ValueError('Shard count and samples per shard must be positive.')
    if arguments.shard_count * arguments.samples_per_shard < MATURE_REPLAY_POSITIONS:
        raise ValueError('The production benchmark must contain at least 2.5 million unique positions.')
    if arguments.world_size <= 0:
        raise ValueError('World size must be positive.')
    if arguments.global_batch_size <= 0 or arguments.global_batch_size % arguments.world_size:
        raise ValueError('Global batch size must be positive and divisible by world size.')
    if arguments.optimizer_steps <= 0 or arguments.quantum_count <= 0:
        raise ValueError('Optimizer-step and quantum counts must be positive.')
    if arguments.trainer_consumption_samples_per_second <= 0:
        raise ValueError('Trainer consumption must be positive.')
    return arguments


def commit_fixture_shard(
    replay_inbox: Path,
    shard_index: int,
    sample_count: int,
) -> None:
    shard_id = f'benchmark-{shard_index:04}'
    hdf5_path = replay_inbox / f'{shard_id}.hdf5'
    write_replay_fixture(
        hdf5_path,
        sample_count=sample_count,
        seed=shard_index,
        state_template_count=8,
    )
    manifest = ReplayShardManifest(
        schema_version=REPLAY_SCHEMA_VERSION,
        shard_id=shard_id,
        game_count=PRODUCTION_GAMES_PER_SHARD,
        unique_sample_count=sample_count,
        producing_worker=shard_index % 16,
        minimum_model_version=100,
        maximum_model_version=102,
        termination_counts=TerminationCounts(
            natural=sample_count,
            resignation=0,
            ply_cap=0,
            material_adjudication=0,
            diagnostic=0,
        ),
        content_sha256=file_sha256(hdf5_path),
        creation_timestamp_seconds=float(shard_index),
        hdf5_file_name=hdf5_path.name,
    )
    (replay_inbox / manifest.manifest_file_name).write_text(
        manifest.model_dump_json(indent=2),
        encoding='utf-8',
    )


def prepare_replay(arguments: Arguments) -> RollingReplayBuffer:
    if arguments.workspace.exists():
        if any(arguments.workspace.iterdir()):
            raise ValueError(f'Benchmark workspace {arguments.workspace} must be nonexistent or empty.')
    else:
        arguments.workspace.mkdir(parents=True)
    replay_inbox = arguments.workspace / 'inbox'
    for shard_index in range(arguments.shard_count):
        commit_fixture_shard(
            replay_inbox=replay_inbox,
            shard_index=shard_index,
            sample_count=arguments.samples_per_shard,
        )
    replay_buffer = RollingReplayBuffer(
        replay_inbox,
        arguments.workspace / 'index.json',
        sampler_seed=arguments.sampler_seed,
    )
    replay_buffer.discover_committed_shards()
    return replay_buffer


def quantum_requests(
    arguments: Arguments,
    rank: int,
) -> tuple[ReplayQuantumRequest, ...]:
    return tuple(
        ReplayQuantumRequest(
            global_step=quantum_index * arguments.optimizer_steps,
            optimizer_steps=arguments.optimizer_steps,
            global_batch_size=arguments.global_batch_size,
            world_size=arguments.world_size,
            rank=rank,
        )
        for quantum_index in range(arguments.quantum_count)
    )


def consume_quantum_views(quantum: ReplayTrainingQuantum) -> int:
    batches = tuple(quantum.optimizer_batches())
    if len(batches) != quantum.optimizer_steps:
        raise RuntimeError('Replay quantum produced the wrong number of optimizer batches.')
    return sum(len(batch) for batch in batches)


def training_batch_bytes(quantum: ReplayTrainingQuantum) -> int:
    batch = quantum.full_batch
    return sum(
        tensor.nelement() * tensor.element_size()
        for tensor in (
            batch.states,
            batch.policy_targets,
            batch.final_outcomes,
            batch.mcts_root_values,
            batch.outcome_target_eligible,
            batch.termination_reasons,
            batch.plies,
            batch.current_player_piece_counts,
            batch.opponent_piece_counts,
        )
    )


def run_decode_worker(
    replay_inbox: Path,
    index_path: Path,
    arguments: Arguments,
    rank: int,
    start_event: Event,
    connection: Connection,
) -> None:
    try:
        replay_buffer = RollingReplayBuffer(
            replay_inbox,
            index_path,
            sampler_seed=arguments.sampler_seed,
            read_only=True,
        )
        connection.send(WorkerReady().model_dump_json())
        start_event.wait()
        started = time.perf_counter()
        decoded_samples = 0
        maximum_resident_bytes = 0
        payload_opens = 0
        selected_rows = 0
        rows_read = 0
        selected_bytes = 0
        bytes_read = 0
        for quantum in prefetch_rank_quanta(replay_buffer, quantum_requests(arguments, rank)):
            decoded_samples += consume_quantum_views(quantum)
            maximum_resident_bytes = max(maximum_resident_bytes, training_batch_bytes(quantum))
            statistics = quantum.decode_statistics
            payload_opens += statistics.payload_open_count
            selected_rows += statistics.selected_rows
            rows_read += statistics.rows_read
            selected_bytes += statistics.selected_bytes
            bytes_read += statistics.bytes_read
        result = ProcessDecodeResult(
            rank=rank,
            decoded_samples=decoded_samples,
            elapsed_seconds=time.perf_counter() - started,
            maximum_resident_quantum_bytes=maximum_resident_bytes,
            payload_opens=payload_opens,
            selected_rows=selected_rows,
            rows_read=rows_read,
            selected_bytes=selected_bytes,
            bytes_read=bytes_read,
        )
        connection.send(WorkerSuccess(result=result).model_dump_json())
    except Exception as exception:
        connection.send(
            WorkerFailure(
                exception_type=type(exception).__name__,
                message=str(exception),
            ).model_dump_json()
        )
    finally:
        connection.close()


def receive_worker_message(connection: Connection, timeout_seconds: float) -> WorkerMessage:
    if not connection.poll(timeout_seconds):
        raise TimeoutError('Replay benchmark worker did not report before the timeout.')
    received_message: str = connection.recv()
    return WORKER_MESSAGE_ADAPTER.validate_json(received_message)


def benchmark_coordinated_processes(
    replay_buffer: RollingReplayBuffer,
    arguments: Arguments,
) -> CoordinatedDecodeBenchmark:
    context = multiprocessing.get_context('spawn')
    start_event = context.Event()
    parent_connections: list[Connection] = []
    processes: list[multiprocessing.Process] = []
    try:
        for rank in range(arguments.world_size):
            parent_connection, child_connection = context.Pipe(duplex=True)
            process = context.Process(
                target=run_decode_worker,
                args=(
                    replay_buffer.replay_inbox,
                    replay_buffer.index_path,
                    arguments,
                    rank,
                    start_event,
                    child_connection,
                ),
                name=f'replay-benchmark-rank-{rank}',
            )
            process.start()
            child_connection.close()
            parent_connections.append(parent_connection)
            processes.append(process)

        for connection in parent_connections:
            message = receive_worker_message(connection, WORKER_READY_TIMEOUT_SECONDS)
            match message:
                case WorkerReady():
                    pass
                case WorkerFailure(exception_type=exception_type, message=message):
                    raise RuntimeError(f'Replay worker failed before synchronization with {exception_type}: {message}')
                case WorkerSuccess():
                    raise RuntimeError('Replay worker decoded before synchronization.')

        started = time.perf_counter()
        start_event.set()
        results: list[ProcessDecodeResult] = []
        for connection in parent_connections:
            message = receive_worker_message(connection, WORKER_RESULT_TIMEOUT_SECONDS)
            match message:
                case WorkerSuccess(result=result):
                    results.append(result)
                case WorkerFailure(exception_type=exception_type, message=message):
                    raise RuntimeError(f'Replay worker failed with {exception_type}: {message}')
                case WorkerReady():
                    raise RuntimeError('Replay worker emitted duplicate readiness.')
        wall_seconds = time.perf_counter() - started

        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                raise TimeoutError(f'Replay benchmark process {process.name} did not exit.')
            if process.exitcode != 0:
                raise RuntimeError(f'Replay benchmark process {process.name} exited with {process.exitcode}.')
    finally:
        start_event.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
        for connection in parent_connections:
            connection.close()

    decoded_samples = sum(result.decoded_samples for result in results)
    selected_rows = sum(result.selected_rows for result in results)
    rows_read = sum(result.rows_read for result in results)
    selected_bytes = sum(result.selected_bytes for result in results)
    bytes_read = sum(result.bytes_read for result in results)
    return CoordinatedDecodeBenchmark(
        wall_seconds=wall_seconds,
        global_samples_per_second=decoded_samples / wall_seconds,
        decoded_samples=decoded_samples,
        maximum_rank_quantum_bytes=max(result.maximum_resident_quantum_bytes for result in results),
        payload_opens=sum(result.payload_opens for result in results),
        selected_rows=selected_rows,
        rows_read=rows_read,
        row_read_amplification=rows_read / selected_rows,
        selected_bytes=selected_bytes,
        bytes_read=bytes_read,
        byte_read_amplification=bytes_read / selected_bytes,
        process_results=tuple(sorted(results, key=lambda result: result.rank)),
    )


def compact_all_producer_shards(replay_buffer: RollingReplayBuffer) -> tuple[int, float]:
    compacted_containers = 0
    started = time.perf_counter()
    while True:
        result = replay_buffer.compact_one_idle_container()
        if result.status is CompactionStepStatus.WAITING_FOR_MORE_SHARDS:
            return compacted_containers, time.perf_counter() - started
        compacted_containers += 1


def main() -> None:
    arguments = parse_arguments()
    replay_buffer = prepare_replay(arguments)
    actual_positions = replay_buffer.unique_sample_count
    producer_shards = replay_buffer.shard_count
    metadata_before = replay_buffer.metadata_memory_bytes
    uncompacted = benchmark_coordinated_processes(replay_buffer, arguments)
    compacted_containers, compaction_seconds = compact_all_producer_shards(replay_buffer)
    compacted = benchmark_coordinated_processes(replay_buffer, arguments)
    result = ReplayLoaderBenchmark(
        replay_capacity_unique_positions=MATURE_REPLAY_POSITIONS,
        actual_unique_positions=actual_positions,
        producer_shards_before_compaction=producer_shards,
        compacted_containers=compacted_containers,
        logical_segments_after_compaction=replay_buffer.shard_count,
        metadata_memory_bytes_before_compaction=metadata_before,
        metadata_memory_bytes_after_compaction=replay_buffer.metadata_memory_bytes,
        assumed_games_per_producer_shard=PRODUCTION_GAMES_PER_SHARD,
        measured_positions_per_game=MEASURED_POSITIONS_PER_GAME,
        compaction_seconds=compaction_seconds,
        compaction_preserved_sample_count=replay_buffer.unique_sample_count == actual_positions,
        global_batch_size=arguments.global_batch_size,
        world_size=arguments.world_size,
        optimizer_steps=arguments.optimizer_steps,
        quantum_count=arguments.quantum_count,
        rank_positions_per_quantum=(arguments.global_batch_size * arguments.optimizer_steps // arguments.world_size),
        trainer_global_consumption_samples_per_second=arguments.trainer_consumption_samples_per_second,
        uncompacted=uncompacted,
        compacted=compacted,
        compacted_throughput_exceeds_consumption=(
            compacted.global_samples_per_second > arguments.trainer_consumption_samples_per_second
        ),
    )
    result_json = json.dumps(asdict(result), indent=2)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(result_json, encoding='utf-8')
    print(result_json)
    if not result.compaction_preserved_sample_count:
        raise RuntimeError('Replay compaction changed the mature buffer sample count.')
    if not result.compacted_throughput_exceeds_consumption:
        raise RuntimeError('Coordinated four-process replay throughput does not exceed trainer consumption.')


if __name__ == '__main__':
    main()
