from __future__ import annotations

import os
import socket
import time
import traceback
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import TypeAlias

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.cluster.CudaProcess import start_process_on_cuda_device
from src.Network import Network
from src.train.ChessReplay import (
    ChessReplayMaintainer,
    ChessReplaySnapshot,
    training_batch_loader,
)
from src.train.Trainer import Trainer, _LogitForward
from src.train.TrainingArgs import ClusterParams, TrainingArgs, TrainingParams
from src.train.TrainingStats import TrainingStats
from src.util.log import configure_logging, log
from src.util.profiler import start_cpu_usage_logger
from src.util.save_paths import checkpoint_manifest_path, load_model_and_optimizer, save_model_and_optimizer


PROCESS_GROUP_TIMEOUT = timedelta(minutes=5)


@dataclass(frozen=True)
class RankReady:
    rank: int


@dataclass(frozen=True)
class RankStopped:
    rank: int


@dataclass(frozen=True)
class RankFailure:
    rank: int
    phase_id: int | None
    exception_type: str
    message: str
    formatted_traceback: str


class DistributedTrainingError(RuntimeError):
    def __init__(self, failure: RankFailure) -> None:
        self.failure = failure
        super().__init__(
            f'DDP rank {failure.rank} failed with {failure.exception_type}: {failure.message}\n'
            f'{failure.formatted_traceback}'
        )


def is_rank_zero(rank: int) -> bool:
    return rank == 0


def validate_distributed_training_configuration(
    cluster: ClusterParams,
    training: TrainingParams,
    cuda_device_count: int,
) -> None:
    device_ids = cluster.trainer_ddp_device_ids
    if not device_ids:
        raise ValueError('At least one DDP trainer device must be configured.')
    if any(device_id < 0 for device_id in device_ids):
        raise ValueError('DDP trainer device IDs cannot be negative.')
    if len(set(device_ids)) != len(device_ids):
        raise ValueError('DDP trainer device IDs must be unique.')
    if device_ids[0] != cluster.trainer_rank_zero_device_id:
        raise ValueError('The rank-zero trainer device must be first in the DDP device list.')
    if training.global_batch_size != training.local_batch_size * len(device_ids):
        raise ValueError('Global batch size must equal local batch size times DDP world size.')
    if cluster.trainer_process_group_backend == 'nccl' and cluster.trainer_device_type != 'cuda':
        raise ValueError('NCCL can only be used with CUDA trainer devices.')
    match cluster.trainer_device_type:
        case 'cuda':
            invalid_device_ids = tuple(device_id for device_id in device_ids if device_id >= cuda_device_count)
            if invalid_device_ids:
                raise ValueError(f'DDP trainer devices {invalid_device_ids} are outside the visible CUDA range.')
        case 'cpu':
            if cluster.trainer_process_group_backend != 'gloo':
                raise ValueError('CPU distributed training requires the Gloo backend.')


def available_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(('127.0.0.1', 0))
        return int(server.getsockname()[1])


def _load_rank_model_and_optimizer(
    args: TrainingArgs,
    model_version: int,
    device: torch.device,
) -> tuple[Network, torch.optim.Optimizer]:
    return load_model_and_optimizer(
        model_version,
        args.network,
        device,
        args.save_path,
        args.training.optimizer,
    )


def _training_device(args: TrainingArgs, rank: int) -> torch.device:
    if args.cluster.trainer_device_type == 'cpu':
        return torch.device('cpu')
    assert rank < len(args.cluster.trainer_ddp_device_ids)
    torch.cuda.set_device(0)
    return torch.device('cuda', 0)


def _wrap_distributed_model(
    model: Network,
    device: torch.device,
) -> DistributedDataParallel:
    logit_model: nn.Module = _LogitForward(model)
    if device.type == 'cuda':
        assert device.index is not None
        return DistributedDataParallel(
            logit_model,
            device_ids=[device.index],
            output_device=device.index,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    return DistributedDataParallel(
        logit_model,
        gradient_as_bucket_view=True,
        static_graph=True,
    )


@dataclass(frozen=True)
class MaintainReplayCommand:
    phase_id: int
    replay_capacity_unique_positions: int

    def __post_init__(self) -> None:
        if self.replay_capacity_unique_positions <= 0:
            raise ValueError('Replay capacity must be positive.')


@dataclass(frozen=True)
class TrainQuantumCommand:
    phase_id: int
    global_step: int
    model_version: int


@dataclass(frozen=True)
class StopTrainerCommand:
    pass


TrainerCommand: TypeAlias = MaintainReplayCommand | TrainQuantumCommand | StopTrainerCommand


@dataclass(frozen=True)
class RankReplayMaintained:
    rank: int
    phase_id: int
    credited_unique_samples: int
    credited_completed_searches: int
    live_unique_samples: int
    compacted_container: bool
    oldest_source_model_version: int | None
    newest_source_model_version: int | None
    weighted_mean_source_model_version_midpoint: float | None
    oldest_position_age_seconds: float | None
    weighted_mean_position_age_seconds: float | None
    evicted_unique_samples: int
    replay_memory_bytes: int


@dataclass(frozen=True)
class RankQuantumComplete:
    rank: int
    phase_id: int
    global_step: int
    model_version: int
    training_stats: TrainingStats | None
    optimizer_seconds: float | None
    decode_seconds: float | None
    transfer_seconds: float | None
    payload_open_count: int | None
    selected_rows: int | None
    rows_read: int | None
    selected_bytes: int | None
    bytes_read: int | None
    checkpoint_manifest: str | None


TrainerResponse: TypeAlias = RankReady | RankReplayMaintained | RankQuantumComplete | RankStopped | RankFailure


@dataclass(frozen=True)
class ReplayState:
    credited_unique_samples: int
    credited_completed_searches: int
    live_unique_samples: int
    compacted_container: bool
    oldest_source_model_version: int | None
    newest_source_model_version: int | None
    weighted_mean_source_model_version_midpoint: float | None
    oldest_position_age_seconds: float | None
    weighted_mean_position_age_seconds: float | None
    evicted_unique_samples: int
    replay_memory_bytes: int


@dataclass(frozen=True)
class QuantumResult:
    global_step: int
    model_version: int
    training_stats: TrainingStats
    optimizer_seconds: float
    decode_seconds: float
    transfer_seconds: float
    payload_open_count: int
    selected_rows: int
    rows_read: int
    selected_bytes: int
    bytes_read: int
    checkpoint_manifest: Path


class TrainerProcess:
    """Persistent DDP ranks for phase-separated replay maintenance and training."""

    def __init__(
        self,
        args: TrainingArgs,
        run_id: int,
        starting_model_version: int,
    ) -> None:
        parameters = args.training.credit_training
        validate_distributed_training_configuration(
            args.cluster,
            args.training,
            torch.cuda.device_count(),
        )
        if len(args.cluster.trainer_ddp_device_ids) != 4:
            raise ValueError('Credit-driven production training requires exactly four DDP ranks.')
        if parameters.maximum_optimizer_steps != 500_000:
            raise ValueError('Credit-driven production training requires a 500,000 optimizer-step limit.')
        if parameters.retained_checkpoint_interval_steps != 1_000:
            raise ValueError('Credit-driven production training requires retained checkpoints every 1,000 steps.')
        self.args = args
        self.run_id = run_id
        self.world_size = len(args.cluster.trainer_ddp_device_ids)
        self._phase_id = 0
        self._context = multiprocessing.get_context('spawn')
        self._connections: list[Connection] = []
        self._processes: list[multiprocessing.Process] = []
        self._closed = False
        self._failed = False
        self._start_workers(starting_model_version)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if not self._failed and self._processes:
                for connection in self._connections:
                    connection.send(StopTrainerCommand())
                self._collect_responses(RankStopped, phase_id=None)
        finally:
            self._terminate_workers(force=False)

    def maintain_replay(
        self,
        replay_capacity_unique_positions: int,
    ) -> ReplayState:
        self._phase_id += 1
        command = MaintainReplayCommand(
            self._phase_id,
            replay_capacity_unique_positions,
        )
        for connection in self._connections:
            connection.send(command)
        responses = self._collect_responses(RankReplayMaintained, command.phase_id)
        credited_counts = {
            response.credited_unique_samples for response in responses if isinstance(response, RankReplayMaintained)
        }
        live_counts = {
            response.live_unique_samples for response in responses if isinstance(response, RankReplayMaintained)
        }
        completed_search_counts = {
            response.credited_completed_searches for response in responses if isinstance(response, RankReplayMaintained)
        }
        if len(credited_counts) != 1 or len(completed_search_counts) != 1 or len(live_counts) != 1:
            raise RuntimeError('DDP ranks did not receive the same frozen chess replay snapshot.')
        rank_zero = responses[0]
        assert isinstance(rank_zero, RankReplayMaintained)
        return ReplayState(
            credited_unique_samples=credited_counts.pop(),
            credited_completed_searches=completed_search_counts.pop(),
            live_unique_samples=live_counts.pop(),
            compacted_container=rank_zero.compacted_container,
            oldest_source_model_version=rank_zero.oldest_source_model_version,
            newest_source_model_version=rank_zero.newest_source_model_version,
            weighted_mean_source_model_version_midpoint=rank_zero.weighted_mean_source_model_version_midpoint,
            oldest_position_age_seconds=rank_zero.oldest_position_age_seconds,
            weighted_mean_position_age_seconds=rank_zero.weighted_mean_position_age_seconds,
            evicted_unique_samples=rank_zero.evicted_unique_samples,
            replay_memory_bytes=rank_zero.replay_memory_bytes,
        )

    def train_quantum(self, global_step: int, model_version: int) -> QuantumResult:
        self._phase_id += 1
        command = TrainQuantumCommand(
            phase_id=self._phase_id,
            global_step=global_step,
            model_version=model_version,
        )
        for connection in self._connections:
            connection.send(command)
        responses = self._collect_responses(RankQuantumComplete, command.phase_id)
        rank_zero = responses[0]
        assert isinstance(rank_zero, RankQuantumComplete)
        if rank_zero.training_stats is None:
            raise RuntimeError('Rank zero did not return credit-quantum training statistics.')
        if rank_zero.optimizer_seconds is None:
            raise RuntimeError('Rank zero did not return credit-quantum optimizer timing.')
        if rank_zero.decode_seconds is None:
            raise RuntimeError('Rank zero did not return credit-quantum replay timing.')
        if rank_zero.transfer_seconds is None:
            raise RuntimeError('Rank zero did not return credit-quantum transfer timing.')
        replay_statistics = (
            rank_zero.payload_open_count,
            rank_zero.selected_rows,
            rank_zero.rows_read,
            rank_zero.selected_bytes,
            rank_zero.bytes_read,
        )
        if any(value is None for value in replay_statistics):
            raise RuntimeError('Rank zero did not return credit-quantum replay statistics.')
        if rank_zero.checkpoint_manifest is None:
            raise RuntimeError('Rank zero did not return the prepared checkpoint manifest.')
        expected_global_step = global_step + self.args.training.credit_training.optimizer_steps_per_quantum
        if any(
            not isinstance(response, RankQuantumComplete)
            or response.global_step != expected_global_step
            or response.model_version != model_version
            for response in responses
        ):
            raise RuntimeError('DDP ranks disagreed about the completed credit quantum.')
        return QuantumResult(
            global_step=rank_zero.global_step,
            model_version=rank_zero.model_version,
            training_stats=rank_zero.training_stats,
            optimizer_seconds=rank_zero.optimizer_seconds,
            decode_seconds=rank_zero.decode_seconds,
            transfer_seconds=rank_zero.transfer_seconds,
            payload_open_count=int(rank_zero.payload_open_count),
            selected_rows=int(rank_zero.selected_rows),
            rows_read=int(rank_zero.rows_read),
            selected_bytes=int(rank_zero.selected_bytes),
            bytes_read=int(rank_zero.bytes_read),
            checkpoint_manifest=Path(rank_zero.checkpoint_manifest),
        )

    def _start_workers(self, starting_model_version: int) -> None:
        initialization_method = f'tcp://127.0.0.1:{available_tcp_port()}'
        for rank in range(self.world_size):
            physical_device_id = self.args.cluster.trainer_ddp_device_ids[rank]
            parent_connection, child_connection = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=run_trainer_rank,
                args=(
                    rank,
                    self.args,
                    self.run_id,
                    starting_model_version,
                    initialization_method,
                    child_connection,
                ),
                name=f'ddp-trainer-rank-{rank}',
            )
            start_process_on_cuda_device(process, physical_device_id)
            child_connection.close()
            self._connections.append(parent_connection)
            self._processes.append(process)
        try:
            self._collect_responses(RankReady, phase_id=None)
        except BaseException:
            self._terminate_workers(force=True)
            raise

    def _collect_responses(
        self,
        expected_type: type[RankReady] | type[RankReplayMaintained] | type[RankQuantumComplete] | type[RankStopped],
        phase_id: int | None,
    ) -> list[TrainerResponse]:
        pending_ranks = set(range(self.world_size))
        responses: list[TrainerResponse | None] = [None] * self.world_size
        while pending_ranks:
            for rank in tuple(pending_ranks):
                connection = self._connections[rank]
                if not connection.poll():
                    continue
                response = connection.recv()
                if not isinstance(
                    response,
                    (
                        RankReady,
                        RankReplayMaintained,
                        RankQuantumComplete,
                        RankStopped,
                        RankFailure,
                    ),
                ):
                    self._terminate_workers(force=True)
                    raise RuntimeError(f'DDP rank {rank} returned an unsupported credit response.')
                if isinstance(response, RankFailure):
                    self._failed = True
                    self._terminate_workers(force=True)
                    raise DistributedTrainingError(response)
                response_phase_id = (
                    response.phase_id if isinstance(response, (RankReplayMaintained, RankQuantumComplete)) else None
                )
                if not isinstance(response, expected_type) or response.rank != rank or response_phase_id != phase_id:
                    self._terminate_workers(force=True)
                    raise RuntimeError(f'DDP rank {rank} returned a stale or unexpected credit response: {response!r}')
                responses[rank] = response
                pending_ranks.remove(rank)

            for rank in tuple(pending_ranks):
                process = self._processes[rank]
                if process.is_alive():
                    continue
                process.join()
                failure = RankFailure(
                    rank=rank,
                    phase_id=phase_id,
                    exception_type='ProcessExit',
                    message=f'credit trainer process exited with code {process.exitcode}',
                    formatted_traceback='',
                )
                self._failed = True
                self._terminate_workers(force=True)
                raise DistributedTrainingError(failure)
            if pending_ranks:
                wait_objects = [self._connections[rank] for rank in pending_ranks]
                wait_objects.extend(self._processes[rank].sentinel for rank in pending_ranks)
                wait(wait_objects, timeout=1)
        return [response for response in responses if response is not None]

    def _terminate_workers(self, force: bool) -> None:
        if force:
            for process in self._processes:
                if process.is_alive():
                    process.terminate()
        for process in self._processes:
            process.join(timeout=10)
        for process in self._processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
        for connection in self._connections:
            connection.close()
        self._connections = []
        self._processes = []


def _maintain_replay(
    command: MaintainReplayCommand,
    replay_maintainer: ChessReplayMaintainer | None,
    rank: int,
    device: torch.device,
) -> tuple[RankReplayMaintained, ChessReplaySnapshot]:
    distributed.barrier()
    snapshot: ChessReplaySnapshot | None = None
    metrics = None
    if is_rank_zero(rank):
        if replay_maintainer is None:
            raise RuntimeError('Rank zero must own the chess replay maintainer.')
        snapshot, metrics = replay_maintainer.maintain(command.replay_capacity_unique_positions)
    objects: list[ChessReplaySnapshot | None] = [snapshot]
    distributed.broadcast_object_list(
        objects,
        src=0,
        device=device,
    )
    received_snapshot = objects[0]
    if received_snapshot is None:
        raise RuntimeError('DDP rank did not receive a frozen chess replay snapshot.')
    if metrics is None:
        samples = received_snapshot.samples
        generations = tuple(sample.source_model_generation for sample in samples)
        ages = tuple(
            max(0.0, received_snapshot.frozen_at_seconds - sample.source_created_at_seconds) for sample in samples
        )
        oldest_generation = min(generations) if generations else None
        newest_generation = max(generations) if generations else None
        mean_generation = sum(generations) / len(generations) if generations else None
        oldest_age = max(ages) if ages else None
        mean_age = sum(ages) / len(ages) if ages else None
    else:
        oldest_generation = metrics.oldest_source_model_generation
        newest_generation = metrics.newest_source_model_generation
        mean_generation = metrics.mean_source_model_generation
        oldest_age = metrics.oldest_sample_age_seconds
        mean_age = metrics.mean_sample_age_seconds
    return (
        RankReplayMaintained(
            rank=rank,
            phase_id=command.phase_id,
            credited_unique_samples=received_snapshot.credited_samples,
            credited_completed_searches=received_snapshot.credited_completed_searches,
            live_unique_samples=len(received_snapshot.samples),
            compacted_container=False,
            oldest_source_model_version=oldest_generation,
            newest_source_model_version=newest_generation,
            weighted_mean_source_model_version_midpoint=mean_generation,
            oldest_position_age_seconds=oldest_age,
            weighted_mean_position_age_seconds=mean_age,
            evicted_unique_samples=received_snapshot.evicted_samples,
            replay_memory_bytes=received_snapshot.estimated_sample_bytes,
        ),
        received_snapshot,
    )


def _train_quantum(
    command: TrainQuantumCommand,
    args: TrainingArgs,
    rank: int,
    model: Network,
    optimizer: torch.optim.Optimizer,
    training_model: DistributedDataParallel,
    replay_snapshot: ChessReplaySnapshot,
) -> RankQuantumComplete:
    parameters = args.training.credit_training
    batches = training_batch_loader(
        replay_snapshot,
        global_step=command.global_step,
        optimizer_steps=parameters.optimizer_steps_per_quantum,
        global_batch_size=args.training.global_batch_size,
        world_size=len(args.cluster.trainer_ddp_device_ids),
        rank=rank,
        pin_memory=model.device.type == 'cuda',
    )
    selected_rows = len(batches.indices)
    if len(batches) != parameters.optimizer_steps_per_quantum:
        raise RuntimeError('Decoded replay quantum has the wrong optimizer-step count.')
    distributed.barrier()

    trainer = Trainer(
        model,
        optimizer,
        args.training,
        training_model=training_model,
        rank=rank,
    )
    if model.device.type == 'cuda':
        torch.cuda.synchronize(model.device)
    optimizer_started_at = time.perf_counter()
    training_stats = trainer.train(
        batches,
        command.global_step,
    )
    decode_seconds = torch.tensor(
        batches.preparation_seconds,
        device=model.device,
        dtype=torch.float64,
    )
    distributed.all_reduce(decode_seconds, op=distributed.ReduceOp.MAX)
    transfer_seconds = torch.tensor(
        trainer.last_transfer_seconds,
        device=model.device,
        dtype=torch.float64,
    )
    distributed.all_reduce(transfer_seconds, op=distributed.ReduceOp.MAX)
    if model.device.type == 'cuda':
        torch.cuda.synchronize(model.device)
    elapsed_seconds = torch.tensor(
        time.perf_counter() - optimizer_started_at,
        device=model.device,
        dtype=torch.float64,
    )
    distributed.all_reduce(elapsed_seconds, op=distributed.ReduceOp.MAX)

    manifest: Path | None = None
    if is_rank_zero(rank):
        save_model_and_optimizer(
            model,
            optimizer,
            command.model_version,
            args.save_path,
        )
        manifest = checkpoint_manifest_path(command.model_version, args.save_path)
        log(
            f'Prepared credit quantum {command.model_version} at optimizer step '
            f'{command.global_step + parameters.optimizer_steps_per_quantum}.'
        )
    distributed.barrier()
    return RankQuantumComplete(
        rank=rank,
        phase_id=command.phase_id,
        global_step=command.global_step + parameters.optimizer_steps_per_quantum,
        model_version=command.model_version,
        training_stats=training_stats if is_rank_zero(rank) else None,
        optimizer_seconds=float(elapsed_seconds.item()) if is_rank_zero(rank) else None,
        decode_seconds=float(decode_seconds.item()) if is_rank_zero(rank) else None,
        transfer_seconds=float(transfer_seconds.item()) if is_rank_zero(rank) else None,
        payload_open_count=0 if is_rank_zero(rank) else None,
        selected_rows=selected_rows * len(args.cluster.trainer_ddp_device_ids) if is_rank_zero(rank) else None,
        rows_read=selected_rows * len(args.cluster.trainer_ddp_device_ids) if is_rank_zero(rank) else None,
        selected_bytes=0 if is_rank_zero(rank) else None,
        bytes_read=0 if is_rank_zero(rank) else None,
        checkpoint_manifest=str(manifest.resolve()) if manifest is not None else None,
    )


def run_trainer_rank(
    rank: int,
    args: TrainingArgs,
    run_id: int,
    starting_model_version: int,
    initialization_method: str,
    connection: Connection,
) -> None:
    configure_logging(enabled=is_rank_zero(rank))
    current_phase_id: int | None = None
    parameters = args.training.credit_training
    replay_maintainer = (
        ChessReplayMaintainer(
            Path(args.save_path),
            capacity=parameters.replay_capacity_for_model_version(starting_model_version),
            sampler_seed=args.random_seed,
        )
        if is_rank_zero(rank)
        else None
    )
    replay_snapshot: ChessReplaySnapshot | None = None
    usage_logger = start_cpu_usage_logger(run_id, 'trainer') if is_rank_zero(rank) else None
    process_group_initialized = False
    normal_shutdown = False
    try:
        trainer_cpu_threads = args.cluster.trainer_cpu_threads
        torch.set_num_threads(trainer_cpu_threads)
        torch.set_num_interop_threads(args.cluster.trainer_interop_threads)
        os.environ['OMP_NUM_THREADS'] = str(trainer_cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(trainer_cpu_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(trainer_cpu_threads)
        device = _training_device(args, rank)
        distributed.init_process_group(
            backend=args.cluster.trainer_process_group_backend,
            init_method=initialization_method,
            rank=rank,
            world_size=len(args.cluster.trainer_ddp_device_ids),
            timeout=PROCESS_GROUP_TIMEOUT,
        )
        process_group_initialized = True
        torch.manual_seed(args.random_seed)
        model, optimizer = _load_rank_model_and_optimizer(
            args,
            starting_model_version,
            device,
        )
        training_model = _wrap_distributed_model(model, device)
        connection.send(RankReady(rank))

        while True:
            command = connection.recv()
            match command:
                case MaintainReplayCommand():
                    current_phase_id = command.phase_id
                    response, replay_snapshot = _maintain_replay(
                        command,
                        replay_maintainer,
                        rank,
                        device,
                    )
                    connection.send(response)
                case TrainQuantumCommand():
                    current_phase_id = command.phase_id
                    if replay_snapshot is None:
                        raise RuntimeError('Training requires a frozen chess replay snapshot.')
                    connection.send(
                        _train_quantum(
                            command,
                            args,
                            rank,
                            model,
                            optimizer,
                            training_model,
                            replay_snapshot,
                        )
                    )
                case StopTrainerCommand():
                    connection.send(RankStopped(rank))
                    normal_shutdown = True
                    break
                case _:
                    raise ValueError(f'Unsupported trainer command: {command!r}')
    except BaseException as exception:
        try:
            connection.send(
                RankFailure(
                    rank=rank,
                    phase_id=current_phase_id,
                    exception_type=type(exception).__name__,
                    message=str(exception),
                    formatted_traceback=traceback.format_exc(),
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if usage_logger is not None:
            usage_logger.stop()
        if process_group_initialized and normal_shutdown:
            distributed.destroy_process_group()
        connection.close()
