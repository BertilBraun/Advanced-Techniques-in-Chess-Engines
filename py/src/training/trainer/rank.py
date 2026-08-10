from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
import time

import torch
import torch.distributed as distributed
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.experiment.configuration import ExperimentConfiguration
from src.games.composition import create_game_implementation
from src.games.implementation import GameImplementation
from src.replay.batch_loader import MappedReplayBatchLoader
from src.training.batch import TrainingModelOutput
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.persistence import load_model_and_optimizer, save_model_and_optimizer
from src.training.network import Network
from src.training.objective import ResolvedTrainingObjective
from src.training.configuration import TrainerTopologyParams
from src.training.trainer.contracts import (
    RankTrainingFailure,
    RankTrainingResult,
    StopTrainerCommand,
    TrainerCommand,
    TrainerStopped,
    TrainQuantumCommand,
)


class DistributedTrainingModel(nn.Module):
    def __init__(self, model: Network) -> None:
        super().__init__()
        self.model = model

    def forward(self, states: torch.Tensor) -> TrainingModelOutput:
        return self.model.training_output(states)


@dataclass(frozen=True)
class _RankRuntime:
    game: GameImplementation
    model: Network
    distributed_model: DistributedDataParallel
    optimizer: torch.optim.Optimizer
    device: torch.device


@dataclass(frozen=True)
class _LossTotals:
    policy: float
    wdl: float
    auxiliary: tuple[float, ...]
    total: float
    gradient_norm: float


@dataclass(frozen=True)
class _DeviceLossTotals:
    policy: torch.Tensor
    wdl: torch.Tensor
    auxiliary: torch.Tensor
    total: torch.Tensor
    gradient_norm: torch.Tensor


def _initialize_rank(
    rank: int,
    world_size: int,
    rendezvous_port: int,
    configuration: ExperimentConfiguration,
    starting_checkpoint: CheckpointReference,
) -> _RankRuntime:
    game = create_game_implementation(configuration)
    topology = configuration.training.topology.trainer
    torch.set_num_threads(topology.cpu_threads)
    torch.set_num_interop_threads(topology.interop_threads)
    device_id = topology.ddp_device_ids[rank]
    device = torch.device('cpu') if topology.device_type == 'cpu' else torch.device('cuda', device_id)
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
    distributed_model = _create_distributed_model(model, topology, device_id)
    return _RankRuntime(game, model, distributed_model, optimizer, device)


def _create_distributed_model(
    model: Network,
    topology: TrainerTopologyParams,
    device_id: int,
) -> DistributedDataParallel:
    return DistributedDataParallel(
        DistributedTrainingModel(model),
        device_ids=None if topology.device_type == 'cpu' else [device_id],
        broadcast_buffers=False,
    )


def _run_rank_commands(
    connection: Connection,
    rank: int,
    world_size: int,
    configuration: ExperimentConfiguration,
    runtime: _RankRuntime,
) -> None:
    while True:
        command: TrainerCommand = connection.recv()
        match command:
            case StopTrainerCommand():
                connection.send(TrainerStopped(rank=rank))
                return
            case TrainQuantumCommand():
                connection.send(
                    train_rank_quantum(
                        rank,
                        world_size,
                        configuration,
                        runtime.game,
                        runtime.model,
                        runtime.distributed_model,
                        runtime.optimizer,
                        runtime.device,
                        command,
                    )
                )


def trainer_rank_main(
    connection: Connection,
    rank: int,
    world_size: int,
    rendezvous_port: int,
    configuration: ExperimentConfiguration,
    starting_checkpoint: CheckpointReference,
) -> None:
    try:
        runtime = _initialize_rank(rank, world_size, rendezvous_port, configuration, starting_checkpoint)
        _run_rank_commands(connection, rank, world_size, configuration, runtime)
    except BaseException as error:
        try:
            connection.send(RankTrainingFailure(rank=rank, error=f'{type(error).__name__}: {error}'))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()
        connection.close()


def _train_batches(
    loader: MappedReplayBatchLoader,
    distributed_model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    uses_cuda: bool,
    maximum_gradient_norm: float,
    objective: ResolvedTrainingObjective,
) -> _DeviceLossTotals:
    totals = _DeviceLossTotals(
        policy=torch.zeros((), device=device),
        wdl=torch.zeros((), device=device),
        auxiliary=torch.zeros(len(objective.auxiliary_loss_weights), device=device),
        total=torch.zeros((), device=device),
        gradient_norm=torch.zeros((), device=device),
    )
    with loader.prefetch(device, uses_cuda) as prefetched_batches:
        for batch in prefetched_batches:
            optimizer.zero_grad(set_to_none=True)
            output = distributed_model(batch.states)
            loss = objective.calculate_loss(output, batch)
            loss.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(distributed_model.parameters(), maximum_gradient_norm)
            optimizer.step()
            totals.policy.add_(loss.policy.detach())
            totals.wdl.add_(loss.wdl.detach())
            totals.total.add_(loss.total.detach())
            totals.gradient_norm.add_(gradient_norm.detach())
            if loss.auxiliary:
                totals.auxiliary.add_(torch.stack(tuple(auxiliary.detach() for auxiliary in loss.auxiliary)))
    return totals


def _resolve_loss_totals(totals: _DeviceLossTotals) -> _LossTotals:
    host_totals = torch.cat(
        (
            torch.stack((totals.policy, totals.wdl, totals.total, totals.gradient_norm)),
            totals.auxiliary,
        )
    ).cpu()
    return _LossTotals(
        policy=float(host_totals[0]),
        wdl=float(host_totals[1]),
        auxiliary=tuple(float(value) for value in host_totals[4:]),
        total=float(host_totals[2]),
        gradient_norm=float(host_totals[3]),
    )


def _save_rank_checkpoint(
    rank: int,
    model: Network,
    optimizer: torch.optim.Optimizer,
    command: TrainQuantumCommand,
    save_path: str,
) -> CheckpointReference | None:
    if rank != 0:
        return None
    generation = command.target_progress.model_generation
    save_model_and_optimizer(model, optimizer, generation, save_path)
    return CheckpointReference.load(Path(save_path), generation)


def train_rank_quantum(
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
        command.target_progress.completed_optimizer_steps - command.source_progress.completed_optimizer_steps
    )
    uses_cuda = configuration.training.topology.trainer.device_type == 'cuda'
    loader = MappedReplayBatchLoader(
        replay=command.replay,
        state=game.state,
        source_optimizer_step=command.source_progress.completed_optimizer_steps,
        optimizer_steps=optimizer_steps,
        global_batch_size=configuration.training.trainer.global_batch_size,
        world_size=world_size,
        rank=rank,
        sampler_seed=configuration.training.random_seed,
        pin_memory=uses_cuda,
    )
    totals = _resolve_loss_totals(
        _train_batches(
            loader,
            ddp,
            optimizer,
            device,
            uses_cuda,
            configuration.training.trainer.max_grad_norm,
            command.parameters.objective,
        )
    )
    checkpoint = _save_rank_checkpoint(rank, model, optimizer, command, configuration.training.save_path)
    distributed.barrier()
    divisor = float(optimizer_steps)
    return RankTrainingResult(
        rank=rank,
        completed_optimizer_steps=command.target_progress.completed_optimizer_steps,
        policy_loss=totals.policy / divisor,
        wdl_loss=totals.wdl / divisor,
        auxiliary_losses=tuple(value / divisor for value in totals.auxiliary),
        total_loss=totals.total / divisor,
        gradient_norm=totals.gradient_norm / divisor,
        replay_rows_read=loader.rows_read,
        replay_read_seconds=loader.read_seconds,
        elapsed_seconds=time.perf_counter() - started_at,
        checkpoint=checkpoint,
    )
