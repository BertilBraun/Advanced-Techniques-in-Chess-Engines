from __future__ import annotations

import os
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import TypeAlias

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch.nn.parallel import DistributedDataParallel

from src.cluster.CudaProcess import start_process_on_cuda_device
from src.cluster.TrainerProcess import (
    PROCESS_GROUP_TIMEOUT,
    DistributedTrainingError,
    RankFailure,
    RankReady,
    RankStopped,
    _load_rank_model_and_optimizer,
    _training_device,
    _wrap_distributed_model,
    available_tcp_port,
    is_rank_zero,
    validate_distributed_training_configuration,
)
from src.self_play.SelfPlayDataset import TrainingBatch
from src.Network import Network
from src.train.RollingReplayBuffer import (
    CompactionStepStatus,
    ReplayQuantumRequest,
    RollingReplayBuffer,
    decode_rank_quantum,
)
from src.train.Trainer import Trainer
from src.train.TrainingArgs import TrainingArgs
from src.train.TrainingStats import TrainingStats
from src.util.log import configure_logging, log
from src.util.profiler import start_cpu_usage_logger
from src.util.save_paths import checkpoint_manifest_path, save_model_and_optimizer


@dataclass(frozen=True)
class MaintainCreditReplayCommand:
    phase_id: int
    compact_below_credited_unique_samples: int | None

    def __post_init__(self) -> None:
        if self.compact_below_credited_unique_samples is not None and self.compact_below_credited_unique_samples <= 0:
            raise ValueError('Replay compaction credit threshold must be positive.')


@dataclass(frozen=True)
class TrainCreditQuantumCommand:
    phase_id: int
    global_step: int
    model_version: int


@dataclass(frozen=True)
class StopCreditTrainerCommand:
    pass


CreditTrainerCommand: TypeAlias = MaintainCreditReplayCommand | TrainCreditQuantumCommand | StopCreditTrainerCommand


@dataclass(frozen=True)
class RankCreditReplayMaintained:
    rank: int
    phase_id: int
    credited_unique_samples: int
    live_unique_samples: int
    compacted_container: bool


@dataclass(frozen=True)
class RankCreditQuantumComplete:
    rank: int
    phase_id: int
    global_step: int
    model_version: int
    training_stats: TrainingStats | None
    optimizer_seconds: float | None
    checkpoint_manifest: str | None


CreditTrainerResponse: TypeAlias = (
    RankReady | RankCreditReplayMaintained | RankCreditQuantumComplete | RankStopped | RankFailure
)


@dataclass(frozen=True)
class CreditReplayState:
    credited_unique_samples: int
    live_unique_samples: int
    compacted_container: bool


@dataclass(frozen=True)
class CreditQuantumResult:
    global_step: int
    model_version: int
    training_stats: TrainingStats
    optimizer_seconds: float
    checkpoint_manifest: Path


@dataclass(frozen=True)
class _QuantumBatchLoader:
    batches: tuple[TrainingBatch, ...]

    def __iter__(self) -> Iterator[TrainingBatch]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


class CreditTrainerProcess:
    """Persistent DDP ranks for phase-separated replay maintenance and training."""

    def __init__(
        self,
        args: TrainingArgs,
        run_id: int,
        starting_model_version: int,
    ) -> None:
        parameters = args.training.credit_training
        if parameters is None:
            raise ValueError('CreditTrainerProcess requires credit-driven training parameters.')
        validate_distributed_training_configuration(
            args.cluster,
            args.training,
            torch.cuda.device_count(),
        )
        if len(args.cluster.trainer_ddp_device_ids) != 4:
            raise ValueError('Credit-driven production training requires exactly four DDP ranks.')
        if args.training.global_batch_size != 1_024 or args.training.local_batch_size != 256:
            raise ValueError('Credit-driven production training requires global/local batches of 1,024/256.')
        if parameters.optimizer_steps_per_quantum != 50:
            raise ValueError('Credit-driven production training requires 50 optimizer steps per quantum.')
        if parameters.maximum_optimizer_steps != 500_000:
            raise ValueError('Credit-driven production training requires a 500,000 optimizer-step limit.')
        if parameters.retained_checkpoint_interval_steps != 1_000:
            raise ValueError('Credit-driven production training requires retained checkpoints every 1,000 steps.')
        replay_root = Path(args.save_path)
        replay_inbox = replay_root / 'replay_inbox'
        replay_index = replay_root / 'rolling-replay-index.json'
        RollingReplayBuffer(
            replay_inbox=replay_inbox,
            index_path=replay_index,
            capacity=args.training.max_buffer_samples,
            sampler_seed=args.random_seed,
        )

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
                    connection.send(StopCreditTrainerCommand())
                self._collect_responses(RankStopped, phase_id=None)
        finally:
            self._terminate_workers(force=False)

    def maintain_replay(
        self,
        compact_below_credited_unique_samples: int | None,
    ) -> CreditReplayState:
        self._phase_id += 1
        command = MaintainCreditReplayCommand(
            self._phase_id,
            compact_below_credited_unique_samples,
        )
        for connection in self._connections:
            connection.send(command)
        responses = self._collect_responses(RankCreditReplayMaintained, command.phase_id)
        credited_counts = {
            response.credited_unique_samples
            for response in responses
            if isinstance(response, RankCreditReplayMaintained)
        }
        live_counts = {
            response.live_unique_samples for response in responses if isinstance(response, RankCreditReplayMaintained)
        }
        if len(credited_counts) != 1 or len(live_counts) != 1:
            raise RuntimeError('DDP ranks did not refresh to the same rolling replay index.')
        rank_zero = responses[0]
        assert isinstance(rank_zero, RankCreditReplayMaintained)
        return CreditReplayState(
            credited_unique_samples=credited_counts.pop(),
            live_unique_samples=live_counts.pop(),
            compacted_container=rank_zero.compacted_container,
        )

    def train_quantum(self, global_step: int, model_version: int) -> CreditQuantumResult:
        self._phase_id += 1
        command = TrainCreditQuantumCommand(
            phase_id=self._phase_id,
            global_step=global_step,
            model_version=model_version,
        )
        for connection in self._connections:
            connection.send(command)
        responses = self._collect_responses(RankCreditQuantumComplete, command.phase_id)
        rank_zero = responses[0]
        assert isinstance(rank_zero, RankCreditQuantumComplete)
        if rank_zero.training_stats is None:
            raise RuntimeError('Rank zero did not return credit-quantum training statistics.')
        if rank_zero.optimizer_seconds is None:
            raise RuntimeError('Rank zero did not return credit-quantum optimizer timing.')
        if rank_zero.checkpoint_manifest is None:
            raise RuntimeError('Rank zero did not return the prepared checkpoint manifest.')
        expected_global_step = global_step + self.args.training.credit_training.optimizer_steps_per_quantum
        if any(
            not isinstance(response, RankCreditQuantumComplete)
            or response.global_step != expected_global_step
            or response.model_version != model_version
            for response in responses
        ):
            raise RuntimeError('DDP ranks disagreed about the completed credit quantum.')
        return CreditQuantumResult(
            global_step=rank_zero.global_step,
            model_version=rank_zero.model_version,
            training_stats=rank_zero.training_stats,
            optimizer_seconds=rank_zero.optimizer_seconds,
            checkpoint_manifest=Path(rank_zero.checkpoint_manifest),
        )

    def _start_workers(self, starting_model_version: int) -> None:
        initialization_method = f'tcp://127.0.0.1:{available_tcp_port()}'
        for rank in range(self.world_size):
            physical_device_id = self.args.cluster.trainer_ddp_device_ids[rank]
            parent_connection, child_connection = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=run_credit_trainer_rank,
                args=(
                    rank,
                    self.args,
                    self.run_id,
                    starting_model_version,
                    initialization_method,
                    child_connection,
                ),
                name=f'credit-ddp-trainer-rank-{rank}',
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
        expected_type: type[RankReady]
        | type[RankCreditReplayMaintained]
        | type[RankCreditQuantumComplete]
        | type[RankStopped],
        phase_id: int | None,
    ) -> list[CreditTrainerResponse]:
        pending_ranks = set(range(self.world_size))
        responses: list[CreditTrainerResponse | None] = [None] * self.world_size
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
                        RankCreditReplayMaintained,
                        RankCreditQuantumComplete,
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
                    response.phase_id
                    if isinstance(response, (RankCreditReplayMaintained, RankCreditQuantumComplete))
                    else None
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


def credit_quantum_request(
    args: TrainingArgs,
    rank: int,
    global_step: int,
) -> ReplayQuantumRequest:
    parameters = args.training.credit_training
    if parameters is None:
        raise ValueError('A credit quantum request requires credit-training parameters.')
    return ReplayQuantumRequest(
        global_step=global_step,
        optimizer_steps=parameters.optimizer_steps_per_quantum,
        global_batch_size=args.training.global_batch_size,
        world_size=len(args.cluster.trainer_ddp_device_ids),
        rank=rank,
    )


def _maintain_replay(
    command: MaintainCreditReplayCommand,
    replay_buffer: RollingReplayBuffer,
    rank: int,
    device: torch.device,
) -> RankCreditReplayMaintained:
    distributed.barrier()
    compacted_container = False
    if is_rank_zero(rank):
        replay_buffer.discover_committed_shards()
        should_compact = (
            command.compact_below_credited_unique_samples is not None
            and replay_buffer.credited_unique_sample_count < command.compact_below_credited_unique_samples
        )
        if should_compact:
            result = replay_buffer.compact_one_idle_container()
            compacted_container = result.status is CompactionStepStatus.COMMITTED_CONTAINER
    distributed.barrier()
    if not is_rank_zero(rank):
        replay_buffer.refresh_index_for_read()
    distributed.barrier()
    counters = torch.tensor(
        [
            replay_buffer.credited_unique_sample_count,
            replay_buffer.unique_sample_count,
        ],
        dtype=torch.int64,
        device=device,
    )
    distributed.broadcast(counters, src=0)
    return RankCreditReplayMaintained(
        rank=rank,
        phase_id=command.phase_id,
        credited_unique_samples=int(counters[0].item()),
        live_unique_samples=int(counters[1].item()),
        compacted_container=compacted_container,
    )


def _train_credit_quantum(
    command: TrainCreditQuantumCommand,
    args: TrainingArgs,
    rank: int,
    model: Network,
    optimizer: torch.optim.Optimizer,
    training_model: DistributedDataParallel,
    replay_buffer: RollingReplayBuffer,
) -> RankCreditQuantumComplete:
    parameters = args.training.credit_training
    assert parameters is not None
    request = credit_quantum_request(args, rank, command.global_step)
    quantum = decode_rank_quantum(replay_buffer, request)
    if quantum.optimizer_steps != parameters.optimizer_steps_per_quantum:
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
        _QuantumBatchLoader(tuple(quantum.optimizer_batches())),
        command.global_step,
    )
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
    return RankCreditQuantumComplete(
        rank=rank,
        phase_id=command.phase_id,
        global_step=command.global_step + parameters.optimizer_steps_per_quantum,
        model_version=command.model_version,
        training_stats=training_stats if is_rank_zero(rank) else None,
        optimizer_seconds=float(elapsed_seconds.item()) if is_rank_zero(rank) else None,
        checkpoint_manifest=str(manifest.resolve()) if manifest is not None else None,
    )


def run_credit_trainer_rank(
    rank: int,
    args: TrainingArgs,
    run_id: int,
    starting_model_version: int,
    initialization_method: str,
    connection: Connection,
) -> None:
    configure_logging(enabled=is_rank_zero(rank))
    current_phase_id: int | None = None
    replay_root = Path(args.save_path)
    replay_buffer = RollingReplayBuffer(
        replay_inbox=replay_root / 'replay_inbox',
        index_path=replay_root / 'rolling-replay-index.json',
        capacity=args.training.max_buffer_samples,
        sampler_seed=args.random_seed,
        read_only=not is_rank_zero(rank),
    )
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
            fresh_optimizer=False,
        )
        training_model = _wrap_distributed_model(model, device)
        connection.send(RankReady(rank))

        while True:
            command = connection.recv()
            match command:
                case MaintainCreditReplayCommand():
                    current_phase_id = command.phase_id
                    connection.send(_maintain_replay(command, replay_buffer, rank, device))
                case TrainCreditQuantumCommand():
                    current_phase_id = command.phase_id
                    connection.send(
                        _train_credit_quantum(
                            command,
                            args,
                            rank,
                            model,
                            optimizer,
                            training_model,
                            replay_buffer,
                        )
                    )
                case StopCreditTrainerCommand():
                    connection.send(RankStopped(rank))
                    normal_shutdown = True
                    break
                case _:
                    raise ValueError(f'Unsupported credit trainer command: {command!r}')
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
