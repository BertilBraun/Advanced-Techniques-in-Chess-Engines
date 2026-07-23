from __future__ import annotations

import os
import socket
import time
import traceback
from collections.abc import Callable
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
from tqdm import tqdm

from src.experiment.artifact_retention import apply_artifact_retention
from src.cluster.CudaProcess import start_process_on_cuda_device
from src.Network import Network
from src.self_play.SelfPlayDataset import SelfPlayDataset, preserve_prebatched_samples
from src.train.DistributedTraining import DistributedTrainingBatchSampler, distributed_epoch_seed
from src.train.LegacyIterationReplayBuffer import LegacyIterationReplayBuffer
from src.train.Trainer import Trainer, _LogitForward
from src.train.TrainingArgs import ClusterParams, TrainingArgs, TrainingParams
from src.train.TrainingStats import TrainingStats
from src.util.log import configure_logging, error, log
from src.util.profiler import start_cpu_usage_logger
from src.util.save_paths import (
    create_optimizer,
    load_checkpoint_manifest,
    load_model,
    load_model_and_optimizer,
    save_model_and_optimizer,
)
from src.util.tensorboard import TensorboardWriter
from src.util.timing import timeit


PROCESS_GROUP_TIMEOUT = timedelta(minutes=5)


def number_of_games_in_iteration(iteration: int, save_path: str) -> int:
    return SelfPlayDataset.load_iteration_stats(save_path, iteration).num_games


@dataclass(frozen=True)
class ReplayUpdate:
    iteration: int
    files: tuple[str, ...]


@dataclass(frozen=True)
class LoadReplayCommand:
    phase_id: int
    current_iteration: int
    window_size: int
    updates: tuple[ReplayUpdate, ...]


@dataclass(frozen=True)
class TrainCommand:
    phase_id: int
    iteration: int


@dataclass(frozen=True)
class StopCommand:
    pass


TrainerCommand: TypeAlias = LoadReplayCommand | TrainCommand | StopCommand


@dataclass(frozen=True)
class RankReady:
    rank: int


@dataclass(frozen=True)
class RankReplayLoaded:
    rank: int
    phase_id: int
    sample_count: int
    game_count: int


@dataclass(frozen=True)
class RankTrainingComplete:
    rank: int
    phase_id: int
    training_stats: TrainingStats | None
    early_stop: bool
    optimizer_seconds: float | None


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


TrainerResponse: TypeAlias = RankReady | RankReplayLoaded | RankTrainingComplete | RankStopped | RankFailure


@dataclass(frozen=True)
class ReplayBufferStatistics:
    num_samples: int
    num_games: int


class DistributedTrainingError(RuntimeError):
    def __init__(self, failure: RankFailure) -> None:
        self.failure = failure
        super().__init__(
            f'DDP rank {failure.rank} failed with {failure.exception_type}: {failure.message}\n'
            f'{failure.formatted_traceback}'
        )


class TrainingEarlyStop(RuntimeError):
    pass


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
            if device_ids != (0,):
                raise ValueError('CPU distributed training uses the single logical device ID 0.')


def available_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(('127.0.0.1', 0))
        return int(server.getsockname()[1])


class TrainerProcess:
    def __init__(self, args: TrainingArgs, run_id: int, starting_iteration: int) -> None:
        validate_distributed_training_configuration(
            args.cluster,
            args.training,
            torch.cuda.device_count(),
        )
        self.args = args
        self.run_id = run_id
        self.starting_iteration = starting_iteration
        self.world_size = len(args.cluster.trainer_ddp_device_ids)
        self.replay_stats = ReplayBufferStatistics(0, 0)
        self.last_optimizer_seconds = 0.0
        self._phase_id = 0
        self._context = multiprocessing.get_context('spawn')
        self._connections: list[Connection] = []
        self._processes: list[multiprocessing.Process] = []
        self._last_replay_snapshot: tuple[int, int, tuple[ReplayUpdate, ...]] | None = None
        self._closed = False
        self._failed = False
        self._start_workers(starting_iteration, fresh_optimizer=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if not self._failed and self._processes:
                for connection in self._connections:
                    connection.send(StopCommand())
                self._collect_responses(RankStopped, phase_id=None)
        finally:
            self._terminate_workers(force=False)

    @timeit
    def train(self, iteration: int) -> TrainingStats:
        while True:
            self._phase_id += 1
            command = TrainCommand(self._phase_id, iteration)
            for connection in self._connections:
                connection.send(command)
            try:
                responses = self._collect_responses(RankTrainingComplete, command.phase_id)
            except DistributedTrainingError as training_error:
                if 'returned nan values' not in training_error.failure.message:
                    raise
                error('Training failed due to NaN values in the model output. Retrying with a fresh optimizer.')
                self._start_workers(iteration, fresh_optimizer=True)
                self._restore_replay_snapshot()
                continue

            rank_zero_response = responses[0]
            assert isinstance(rank_zero_response, RankTrainingComplete)
            if rank_zero_response.training_stats is None:
                raise RuntimeError('Rank zero did not return training statistics.')
            if any(
                not isinstance(response, RankTrainingComplete) or response.early_stop != rank_zero_response.early_stop
                for response in responses
            ):
                raise RuntimeError('DDP ranks disagreed about the early-stop decision.')
            if rank_zero_response.early_stop:
                raise TrainingEarlyStop('Training stopped early due to low value standard deviation.')
            if rank_zero_response.optimizer_seconds is None:
                raise RuntimeError('Rank zero did not return optimizer timing.')
            self.last_optimizer_seconds = rank_zero_response.optimizer_seconds
            return rank_zero_response.training_stats

    @timeit
    def wait_for_enough_training_samples(
        self,
        iteration: int,
        stop_reason: Callable[[], str | None],
    ) -> bool:
        def games(target_iteration: int) -> int:
            return number_of_games_in_iteration(target_iteration, self.args.save_path)

        target_games = self.args.num_games_per_iteration
        current_games = games(iteration) + 0.5 * games(iteration - 1)
        with tqdm(
            total=target_games, desc=f'Waiting for games (iter {iteration})', initial=int(current_games)
        ) as progress:
            while current_games < target_games:
                if stop_reason() is not None:
                    return False
                time.sleep(1)
                new_games = games(iteration) + 0.5 * games(iteration - 1)
                if new_games > current_games:
                    progress.update(int(min(new_games, target_games) - current_games))
                    current_games = new_games
        return True

    @timeit
    def load_all_memories_to_train_on_for_iteration(self, iteration: int) -> None:
        window_size = self.args.training.sampling_window(iteration)
        first_iteration = max(iteration - window_size + 1, 0)
        updates = tuple(
            ReplayUpdate(
                replay_iteration,
                tuple(
                    str(path.resolve())
                    for path in sorted(
                        SelfPlayDataset.get_files_to_load_for_iteration(
                            self.args.save_path,
                            replay_iteration,
                        )
                    )
                ),
            )
            for replay_iteration in range(first_iteration, iteration + 1)
        )
        log(
            f'Loading replay snapshot for iteration {iteration} with window size {window_size} '
            f'({first_iteration}-{iteration})'
        )

        self._phase_id += 1
        command = LoadReplayCommand(
            phase_id=self._phase_id,
            current_iteration=iteration,
            window_size=window_size,
            updates=updates,
        )
        self._last_replay_snapshot = (iteration, window_size, updates)
        self._send_replay_command(command)

    def _restore_replay_snapshot(self) -> None:
        if self._last_replay_snapshot is None:
            raise RuntimeError('Cannot retry DDP training before loading a replay snapshot.')
        iteration, window_size, updates = self._last_replay_snapshot
        self._phase_id += 1
        self._send_replay_command(
            LoadReplayCommand(
                phase_id=self._phase_id,
                current_iteration=iteration,
                window_size=window_size,
                updates=updates,
            )
        )

    def _send_replay_command(self, command: LoadReplayCommand) -> None:
        for connection in self._connections:
            connection.send(command)
        responses = self._collect_responses(RankReplayLoaded, command.phase_id)
        sample_counts = {response.sample_count for response in responses if isinstance(response, RankReplayLoaded)}
        game_counts = {response.game_count for response in responses if isinstance(response, RankReplayLoaded)}
        if len(sample_counts) != 1 or len(game_counts) != 1:
            raise RuntimeError('DDP ranks loaded different replay-buffer snapshots.')
        self.replay_stats = ReplayBufferStatistics(sample_counts.pop(), game_counts.pop())
        log(f'Loaded {self.replay_stats.num_samples} samples from {self.replay_stats.num_games} games per rank')

    def _start_workers(self, starting_iteration: int, fresh_optimizer: bool) -> None:
        if self._processes:
            self._terminate_workers(force=True)
        self._failed = False
        self._closed = False
        self._connections = []
        self._processes = []
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
                    starting_iteration,
                    fresh_optimizer,
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
        expected_type: type[RankReady] | type[RankReplayLoaded] | type[RankTrainingComplete] | type[RankStopped],
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
                    response, (RankReady, RankReplayLoaded, RankTrainingComplete, RankStopped, RankFailure)
                ):
                    self._terminate_workers(force=True)
                    raise RuntimeError(f'DDP rank {rank} returned an unsupported response.')
                if isinstance(response, RankFailure):
                    self._failed = True
                    self._terminate_workers(force=True)
                    raise DistributedTrainingError(response)
                response_phase_id = (
                    response.phase_id if isinstance(response, (RankReplayLoaded, RankTrainingComplete)) else None
                )
                if not isinstance(response, expected_type) or response.rank != rank or response_phase_id != phase_id:
                    self._terminate_workers(force=True)
                    raise RuntimeError(f'DDP rank {rank} returned a stale or unexpected response: {response!r}')
                responses[rank] = response
                pending_ranks.remove(rank)

            for rank in tuple(pending_ranks):
                process = self._processes[rank]
                if not process.is_alive():
                    process.join()
                    failure = RankFailure(
                        rank=rank,
                        phase_id=phase_id,
                        exception_type='ProcessExit',
                        message=f'trainer process exited with code {process.exitcode}',
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


def _load_rank_model_and_optimizer(
    args: TrainingArgs,
    iteration: int,
    device: torch.device,
    fresh_optimizer: bool,
) -> tuple[Network, torch.optim.Optimizer]:
    if not fresh_optimizer:
        return load_model_and_optimizer(
            iteration,
            args.network,
            device,
            args.save_path,
            args.training.optimizer,
        )
    manifest = load_checkpoint_manifest(iteration, args.save_path)
    model = load_model(Path(args.save_path) / manifest.model_path, args.network, device)
    return model, create_optimizer(model, args.training.optimizer)


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


def _load_replay(
    command: LoadReplayCommand,
    rolling_buffer: LegacyIterationReplayBuffer,
    rank: int,
    run_id: int,
) -> RankReplayLoaded:
    for update in command.updates:
        rolling_buffer.update(
            update.iteration,
            command.window_size,
            [Path(file_path) for file_path in update.files],
        )
    if rank == 0:
        rolling_buffer.log_all_dataset_stats(run_id)
    stats = rolling_buffer.stats
    return RankReplayLoaded(
        rank=rank,
        phase_id=command.phase_id,
        sample_count=stats.num_samples,
        game_count=stats.num_games,
    )


def as_distributed_dataloader(
    dataset: LegacyIterationReplayBuffer,
    args: TrainingArgs,
    rank: int,
    iteration: int,
    epoch: int,
) -> torch.utils.data.DataLoader:
    sampler = DistributedTrainingBatchSampler(
        dataset=dataset,
        global_batch_size=args.training.global_batch_size,
        local_batch_size=args.training.local_batch_size,
        rank=rank,
        world_size=len(args.cluster.trainer_ddp_device_ids),
        seed=distributed_epoch_seed(args.random_seed, iteration, epoch),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.training.num_workers,
        collate_fn=preserve_prebatched_samples,
        persistent_workers=False,
        pin_memory=args.cluster.trainer_device_type == 'cuda',
        prefetch_factor=16 if args.training.num_workers > 0 else None,
        multiprocessing_context='fork' if args.training.num_workers > 0 and os.name != 'nt' else None,
    )


def as_dataloader(
    dataset: SelfPlayDataset | LegacyIterationReplayBuffer,
    batch_size: int,
    num_workers: int,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=drop_last,
        collate_fn=preserve_prebatched_samples,
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=16 if num_workers > 0 else None,
        multiprocessing_context='fork' if num_workers > 0 and os.name != 'nt' else None,
    )


def _train_iteration(
    command: TrainCommand,
    args: TrainingArgs,
    run_id: int,
    rank: int,
    model: Network,
    optimizer: torch.optim.Optimizer,
    training_model: DistributedDataParallel,
    rolling_buffer: LegacyIterationReplayBuffer,
) -> RankTrainingComplete:
    trainer = Trainer(model, optimizer, args.training, training_model=training_model, rank=rank)
    epoch_stats: list[TrainingStats] = []
    optimizer_seconds = 0.0
    early_stop = False
    writer = TensorboardWriter(run_id, 'trainer', postfix_pid=False, enabled=is_rank_zero(rank))
    with writer:
        for epoch in range(args.training.num_epochs):
            dataloader = as_distributed_dataloader(rolling_buffer, args, rank, command.iteration, epoch)
            if model.device.type == 'cuda':
                torch.cuda.synchronize(model.device)
            optimizer_started_at = time.perf_counter()
            training_stats = trainer.train(dataloader, command.iteration)
            if model.device.type == 'cuda':
                torch.cuda.synchronize(model.device)
            elapsed_seconds = torch.tensor(
                time.perf_counter() - optimizer_started_at,
                device=model.device,
                dtype=torch.float64,
            )
            distributed.all_reduce(elapsed_seconds, op=distributed.ReduceOp.MAX)
            optimizer_seconds += float(elapsed_seconds.item())
            epoch_stats.append(training_stats)
            early_stop = training_stats.value_std < 0.01
            if early_stop:
                break

            stop_after_epoch = torch.zeros(1, device=model.device, dtype=torch.int64)
            if rank == 0:
                log('Train stats: ', training_stats)
                save_model_and_optimizer(model, optimizer, command.iteration + 1, args.save_path)
                retention_result = apply_artifact_retention(
                    Path(args.save_path),
                    command.iteration + 1,
                    args.artifact_retention,
                )
                log('Artifact retention:', retention_result)
                if number_of_games_in_iteration(command.iteration, args.save_path) >= args.num_games_per_iteration * 2:
                    stop_after_epoch.fill_(1)
            distributed.broadcast(stop_after_epoch, src=0)
            if stop_after_epoch.item():
                if rank == 0:
                    log('Enough games played, stopping training for this iteration.')
                break

        combined_stats = TrainingStats.combine(epoch_stats)
        if rank == 0:
            combined_stats.log_to_tensorboard(command.iteration, 'train')
    return RankTrainingComplete(
        rank=rank,
        phase_id=command.phase_id,
        training_stats=combined_stats if rank == 0 else None,
        early_stop=early_stop,
        optimizer_seconds=optimizer_seconds if rank == 0 else None,
    )


def run_trainer_rank(
    rank: int,
    args: TrainingArgs,
    run_id: int,
    starting_iteration: int,
    fresh_optimizer: bool,
    initialization_method: str,
    connection: Connection,
) -> None:
    configure_logging(enabled=is_rank_zero(rank))
    current_phase_id: int | None = None
    rolling_buffer = LegacyIterationReplayBuffer(max_buffer_samples=args.training.max_buffer_samples)
    usage_logger = start_cpu_usage_logger(run_id, 'trainer') if rank == 0 else None
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
            starting_iteration,
            device,
            fresh_optimizer,
        )
        training_model = _wrap_distributed_model(model, device)
        connection.send(RankReady(rank))

        while True:
            command = connection.recv()
            match command:
                case LoadReplayCommand():
                    current_phase_id = command.phase_id
                    connection.send(_load_replay(command, rolling_buffer, rank, run_id))
                case TrainCommand():
                    current_phase_id = command.phase_id
                    connection.send(
                        _train_iteration(
                            command,
                            args,
                            run_id,
                            rank,
                            model,
                            optimizer,
                            training_model,
                            rolling_buffer,
                        )
                    )
                case StopCommand():
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
        return
    finally:
        rolling_buffer.close()
        if usage_logger is not None:
            usage_logger.stop()
        if process_group_initialized and normal_shutdown:
            distributed.destroy_process_group()
        connection.close()
