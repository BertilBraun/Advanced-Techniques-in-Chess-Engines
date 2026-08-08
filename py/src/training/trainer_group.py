from __future__ import annotations

from dataclasses import dataclass
import multiprocessing
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
import socket
import time
from typing import Literal, TypeAlias

import torch
import torch.distributed as distributed
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.experiment.configuration import ExperimentConfiguration
from src.games.implementation import GameImplementation
from src.games.composition import create_game_implementation
from src.neural_network import Network
from src.replay.batch_loader import MappedReplayBatchLoader
from src.replay.manager import ReplayDescription
from src.training.batch import TrainingModelOutput
from src.training.checkpoint import CheckpointReference
from src.training.progress import TrainingProgress
from src.training.objective import ResolvedTrainingObjective
from src.util.frozen_model import FrozenModel
from src.util.save_paths import load_model_and_optimizer, save_model_and_optimizer


class ResolvedTrainingParameters(FrozenModel):
    learning_rate: float
    objective: ResolvedTrainingObjective


class TrainQuantumCommand(FrozenModel):
    kind: Literal['train_quantum'] = 'train_quantum'
    replay: ReplayDescription
    source_progress: TrainingProgress
    target_progress: TrainingProgress
    parameters: ResolvedTrainingParameters


class StopTrainerCommand(FrozenModel):
    kind: Literal['stop'] = 'stop'


TrainerCommand: TypeAlias = TrainQuantumCommand | StopTrainerCommand


class RankTrainingResult(FrozenModel):
    kind: Literal['trained'] = 'trained'
    rank: int
    completed_optimizer_steps: int
    policy_loss: float
    wdl_loss: float
    auxiliary_losses: tuple[float, ...]
    total_loss: float
    gradient_norm: float
    replay_rows_read: int
    replay_read_seconds: float
    elapsed_seconds: float
    checkpoint: CheckpointReference | None


class RankTrainingFailure(FrozenModel):
    kind: Literal['failed'] = 'failed'
    rank: int
    error: str


class TrainerStopped(FrozenModel):
    kind: Literal['stopped'] = 'stopped'
    rank: int


TrainerResponse: TypeAlias = RankTrainingResult | RankTrainingFailure | TrainerStopped


@dataclass(frozen=True)
class TrainingStatistics:
    policy_loss: float
    wdl_loss: float
    auxiliary_losses: tuple[float, ...]
    total_loss: float
    gradient_norm: float
    replay_rows_per_second: float
    training_samples_per_second: float
    elapsed_seconds: float


@dataclass(frozen=True)
class TrainingQuantumResult:
    completed_optimizer_steps: int
    checkpoint: CheckpointReference
    statistics: TrainingStatistics


class _DistributedTrainingModel(nn.Module):
    def __init__(self, model: Network) -> None:
        super().__init__()
        self.model = model

    def forward(self, states: torch.Tensor) -> TrainingModelOutput:
        return self.model.training_output(states)


class TrainerGroup:
    def __init__(
        self,
        configuration: ExperimentConfiguration,
        game: GameImplementation,
        starting_checkpoint: CheckpointReference,
    ) -> None:
        self.configuration = configuration
        self.game = game
        self.starting_checkpoint = starting_checkpoint
        topology = configuration.training.topology.trainer
        self.world_size = len(topology.ddp_device_ids)
        self._closed = False
        context = multiprocessing.get_context('spawn')
        rendezvous_port = _available_tcp_port()
        self._connections: list[Connection] = []
        self._processes: list[BaseProcess] = []
        for rank in range(self.world_size):
            parent, child = context.Pipe(duplex=True)
            process = context.Process(
                target=_trainer_rank_main,
                args=(
                    child,
                    rank,
                    self.world_size,
                    rendezvous_port,
                    configuration,
                    starting_checkpoint,
                ),
                name=f'trainer-rank-{rank}',
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)

    def train_quantum(
        self,
        replay: ReplayDescription,
        progress: TrainingProgress,
    ) -> TrainingQuantumResult:
        self._ensure_open()
        source_generation = progress.model_generation
        target_progress = progress.next_generation
        command = TrainQuantumCommand(
            replay=replay,
            source_progress=progress,
            target_progress=target_progress,
            parameters=ResolvedTrainingParameters(
                learning_rate=self.configuration.training.trainer.learning_rate.value_at(
                    source_generation
                ),
                objective=self.game.training_objective_at(source_generation),
            ),
        )
        for connection in self._connections:
            connection.send(command)
        responses = self._receive_quantum_responses()
        results = list(responses)
        if len(results) != self.world_size:
            raise RuntimeError('Trainer ranks returned an invalid response set.')
        expected_steps = target_progress.completed_optimizer_steps
        if any(result.completed_optimizer_steps != expected_steps for result in results):
            raise RuntimeError('Trainer ranks disagree about completed optimizer steps.')
        rank_zero = results[0]
        if rank_zero.rank != 0 or rank_zero.checkpoint is None:
            raise RuntimeError('Rank zero did not return the completed checkpoint.')
        if any(result.checkpoint is not None for result in results[1:]):
            raise RuntimeError('Only rank zero may return a checkpoint.')
        auxiliary_count = len(rank_zero.auxiliary_losses)
        if any(len(result.auxiliary_losses) != auxiliary_count for result in results):
            raise RuntimeError('Trainer ranks disagree about auxiliary statistics.')
        return TrainingQuantumResult(
            completed_optimizer_steps=expected_steps,
            checkpoint=rank_zero.checkpoint,
            statistics=TrainingStatistics(
                policy_loss=_mean(tuple(result.policy_loss for result in results)),
                wdl_loss=_mean(tuple(result.wdl_loss for result in results)),
                auxiliary_losses=tuple(
                    _mean(tuple(result.auxiliary_losses[index] for result in results))
                    for index in range(auxiliary_count)
                ),
                total_loss=_mean(tuple(result.total_loss for result in results)),
                gradient_norm=_mean(tuple(result.gradient_norm for result in results)),
                replay_rows_per_second=(
                    sum(result.replay_rows_read for result in results)
                    / max(result.replay_read_seconds for result in results)
                ),
                training_samples_per_second=(
                    self.configuration.training.trainer.global_batch_size
                    * self.configuration.training.lifecycle.credit.optimizer_steps_per_quantum
                    / max(result.elapsed_seconds for result in results)
                ),
                elapsed_seconds=max(result.elapsed_seconds for result in results),
            ),
        )

    def close(self) -> None:
        if self._closed:
            return
        for connection in self._connections:
            connection.send(StopTrainerCommand())
        for connection in self._connections:
            response = self._receive(connection)
            match response:
                case TrainerStopped():
                    pass
                case _:
                    raise RuntimeError('Trainer rank did not acknowledge shutdown.')
        for process in self._processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f'Trainer rank exited with code {process.exitcode}.')
        for connection in self._connections:
            connection.close()
        self._closed = True

    def _receive_quantum_responses(self) -> tuple[RankTrainingResult, ...]:
        pending = set(self._connections)
        results: list[RankTrainingResult] = []
        while pending:
            for connection in wait(pending):
                try:
                    response = self._receive(connection)
                except RuntimeError:
                    self._terminate_after_rank_failure()
                    raise
                pending.remove(connection)
                match response:
                    case RankTrainingResult():
                        results.append(response)
                    case RankTrainingFailure():
                        self._terminate_after_rank_failure()
                        raise RuntimeError(
                            f'DDP training quantum failed on rank {response.rank}: {response.error}'
                        )
                    case TrainerStopped():
                        self._terminate_after_rank_failure()
                        raise RuntimeError('Trainer rank stopped during a training quantum.')
        return tuple(sorted(results, key=lambda result: result.rank))

    def _terminate_after_rank_failure(self) -> None:
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join()
        for connection in self._connections:
            connection.close()
        self._closed = True

    @staticmethod
    def _receive(connection: Connection) -> TrainerResponse:
        try:
            response: TrainerResponse = connection.recv()
        except EOFError as error:
            raise RuntimeError('Trainer rank connection closed unexpectedly.') from error
        return response

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError('Trainer group is closed.')


def _trainer_rank_main(
    connection: Connection,
    rank: int,
    world_size: int,
    rendezvous_port: int,
    configuration: ExperimentConfiguration,
    starting_checkpoint: CheckpointReference,
) -> None:
    try:
        game = create_game_implementation(configuration)
        topology = configuration.training.topology.trainer
        torch.set_num_threads(topology.cpu_threads)
        torch.set_num_interop_threads(topology.interop_threads)
        device_id = topology.ddp_device_ids[rank]
        device = (
            torch.device('cpu')
            if topology.device_type == 'cpu'
            else torch.device('cuda', device_id)
        )
        if topology.device_type == 'cuda':
            torch.cuda.set_device(device)
        distributed.init_process_group(
            backend=topology.process_group_backend,
            init_method=f'tcp://127.0.0.1:{rendezvous_port}',
            rank=rank,
            world_size=world_size,
        )
        auxiliary_sizes = tuple(head.action_size for head in game.target_layout.auxiliary_heads)
        model, optimizer = load_model_and_optimizer(
            starting_checkpoint.generation,
            configuration.training.network,
            device,
            configuration.training.save_path,
            configuration.training.trainer.optimizer,
            game.network_dimensions,
            auxiliary_sizes,
        )
        training_model = _DistributedTrainingModel(model)
        ddp = DistributedDataParallel(
            training_model,
            device_ids=None if topology.device_type == 'cpu' else [device_id],
        )
        while True:
            command: TrainerCommand = connection.recv()
            match command:
                case StopTrainerCommand():
                    connection.send(TrainerStopped(rank=rank))
                    break
                case TrainQuantumCommand():
                    connection.send(
                        _train_rank_quantum(
                            rank,
                            world_size,
                            configuration,
                            game,
                            model,
                            ddp,
                            optimizer,
                            device,
                            command,
                        )
                    )
    except BaseException as error:
        try:
            connection.send(
                RankTrainingFailure(rank=rank, error=f'{type(error).__name__}: {error}')
            )
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()
        connection.close()


def _train_rank_quantum(
    rank: int,
    world_size: int,
    configuration: ExperimentConfiguration,
    game: GameImplementation,
    model: Network,
    ddp: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    command: TrainQuantumCommand,
) -> RankTrainingResult:
    started_at = time.perf_counter()
    for group in optimizer.param_groups:
        group['lr'] = command.parameters.learning_rate
    optimizer_steps = (
        command.target_progress.completed_optimizer_steps
        - command.source_progress.completed_optimizer_steps
    )
    loader = MappedReplayBatchLoader(
        replay=command.replay,
        state=game.state,
        source_optimizer_step=command.source_progress.completed_optimizer_steps,
        optimizer_steps=optimizer_steps,
        global_batch_size=configuration.training.trainer.global_batch_size,
        world_size=world_size,
        rank=rank,
        sampler_seed=configuration.training.random_seed,
        pin_memory=topology_uses_cuda(configuration),
    )
    policy_losses = 0.0
    wdl_losses = 0.0
    auxiliary_losses = [0.0] * len(command.parameters.objective.auxiliary_loss_weights)
    total_losses = 0.0
    gradient_norms = 0.0
    for batch in loader:
        batch = batch.to_device(device, non_blocking=topology_uses_cuda(configuration))
        optimizer.zero_grad(set_to_none=True)
        output = ddp(batch.states)
        loss = command.parameters.objective.calculate_loss(output, batch)
        loss.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            ddp.parameters(),
            configuration.training.trainer.max_grad_norm,
        )
        optimizer.step()
        policy_losses += float(loss.policy.detach())
        wdl_losses += float(loss.wdl.detach())
        total_losses += float(loss.total.detach())
        gradient_norms += float(gradient_norm.detach())
        for index, auxiliary in enumerate(loss.auxiliary):
            auxiliary_losses[index] += float(auxiliary.detach())
    checkpoint = None
    if rank == 0:
        generation = command.target_progress.model_generation
        save_model_and_optimizer(model, optimizer, generation, configuration.training.save_path)
        checkpoint = CheckpointReference.load(Path(configuration.training.save_path), generation)
    distributed.barrier()
    divisor = float(optimizer_steps)
    return RankTrainingResult(
        rank=rank,
        completed_optimizer_steps=command.target_progress.completed_optimizer_steps,
        policy_loss=policy_losses / divisor,
        wdl_loss=wdl_losses / divisor,
        auxiliary_losses=tuple(value / divisor for value in auxiliary_losses),
        total_loss=total_losses / divisor,
        gradient_norm=gradient_norms / divisor,
        replay_rows_read=loader.rows_read,
        replay_read_seconds=loader.read_seconds,
        elapsed_seconds=time.perf_counter() - started_at,
        checkpoint=checkpoint,
    )


def topology_uses_cuda(configuration: ExperimentConfiguration) -> bool:
    return configuration.training.topology.trainer.device_type == 'cuda'


def _available_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        return int(listener.getsockname()[1])


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError('Cannot average an empty value set.')
    return sum(values) / len(values)
