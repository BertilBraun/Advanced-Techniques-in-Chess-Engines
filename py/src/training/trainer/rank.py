from __future__ import annotations

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


def trainer_rank_main(
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
        distributed_model = DistributedTrainingModel(model)
        ddp = DistributedDataParallel(
            distributed_model,
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
                        train_rank_quantum(
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
            connection.send(RankTrainingFailure(rank=rank, error=f'{type(error).__name__}: {error}'))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()
        connection.close()


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
    policy_losses = 0.0
    wdl_losses = 0.0
    auxiliary_losses = [0.0] * len(command.parameters.objective.auxiliary_loss_weights)
    total_losses = 0.0
    gradient_norms = 0.0
    for batch in loader:
        batch = batch.to_device(device, non_blocking=uses_cuda)
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
