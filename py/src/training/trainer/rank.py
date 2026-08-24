from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path

import torch
import torch.distributed as distributed
from src.evaluation.dataset import load_dataset_probe_states
from src.evaluation.process import resolve_project_path
from src.experiment.configuration import ExperimentConfiguration, load_experiment_configuration_json
from src.games.composition import create_game_implementation
from src.games.implementation import GameImplementation
from src.replay.batch_loader import MappedReplayBatchLoader
from src.training.batch import TrainingModelOutput
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.paths import checkpoint_manifest_path
from src.training.checkpoint.persistence import load_model_and_optimizer, save_model_and_optimizer
from src.training.configuration import TrainerTopologyParams, TrainingCompilation, TrainingPrecision
from src.training.distributions import TrainingDistributionSnapshot, capture_training_distributions
from src.training.network import POLICY_PRIOR_PROBE_POSITIONS, Network
from src.training.objective import ResolvedTrainingObjective
from src.training.trainer.contracts import (
    RankTrainingFailure,
    RankTrainingResult,
    StopTrainerCommand,
    TrainerCommand,
    TrainerStartup,
    TrainerStopped,
    TrainQuantumCommand,
)
from torch import nn
from torch.nn.parallel import DistributedDataParallel


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
    save_path: Path


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


@dataclass(frozen=True)
class _TrainingBatchResult:
    totals: _DeviceLossTotals
    distributions: TrainingDistributionSnapshot | None


def _initialize_rank(
    rank: int,
    world_size: int,
    rendezvous_port: int,
    configuration: ExperimentConfiguration,
    startup: TrainerStartup,
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
    initial_checkpoint_exists = checkpoint_manifest_path(startup.starting_generation, startup.save_path).exists()
    model, optimizer = load_model_and_optimizer(
        startup.starting_generation,
        startup.network,
        device,
        startup.save_path,
        configuration.training.trainer.optimizer,
        game.network_dimensions,
        game.target_layout.auxiliary_heads,
    )
    distributed_model = _create_distributed_model(
        model,
        topology,
        configuration.training.trainer.compilation,
        device_id,
    )
    if not initial_checkpoint_exists:
        if rank == 0:
            bootstrap_probe_states = None
            if startup.starting_generation == 0:
                bootstrap_probe_states = load_dataset_probe_states(
                    resolve_project_path(configuration.evaluation.dataset.path),
                    game.state,
                    POLICY_PRIOR_PROBE_POSITIONS,
                )
            save_model_and_optimizer(
                model,
                optimizer,
                startup.starting_generation,
                startup.save_path,
                bootstrap_probe_states,
            )
        distributed.barrier()
    return _RankRuntime(game, model, distributed_model, optimizer, device, startup.save_path)


def _create_distributed_model(
    model: Network,
    topology: TrainerTopologyParams,
    compilation: TrainingCompilation,
    device_id: int,
) -> DistributedDataParallel:
    training_model: nn.Module = DistributedTrainingModel(model)
    if compilation is TrainingCompilation.DEFAULT:
        training_model = torch.compile(training_model)
    return DistributedDataParallel(
        training_model,
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
                        runtime.save_path,
                        command,
                    )
                )


def trainer_rank_main(
    connection: Connection,
    rank: int,
    world_size: int,
    rendezvous_port: int,
    configuration_json: str,
    startup: TrainerStartup,
) -> None:
    try:
        configuration = load_experiment_configuration_json(configuration_json)
        runtime = _initialize_rank(
            rank,
            world_size,
            rendezvous_port,
            configuration,
            startup,
        )
        _run_rank_commands(connection, rank, world_size, configuration, runtime)
    except BaseException:
        try:
            connection.send(RankTrainingFailure(rank=rank, error=traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()
        connection.close()


def warmup_scaled_learning_rate(
    base_learning_rate: float,
    warmup_optimizer_steps: int,
    completed_optimizer_steps: int,
) -> float:
    if warmup_optimizer_steps <= 0 or completed_optimizer_steps >= warmup_optimizer_steps:
        return base_learning_rate
    return base_learning_rate * (completed_optimizer_steps + 1) / warmup_optimizer_steps


def _train_batches(
    loader: MappedReplayBatchLoader,
    distributed_model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    uses_cuda: bool,
    maximum_gradient_norm: float,
    objective: ResolvedTrainingObjective,
    collect_distributions: bool,
    source_generation: int,
    precision: TrainingPrecision,
    base_learning_rate: float,
    warmup_optimizer_steps: int,
    completed_optimizer_steps: int,
    replay_prefetch_depth: int,
) -> _TrainingBatchResult:
    totals = _DeviceLossTotals(
        policy=torch.zeros((), device=device),
        wdl=torch.zeros((), device=device),
        auxiliary=torch.zeros(len(objective.auxiliary_losses), device=device),
        total=torch.zeros((), device=device),
        gradient_norm=torch.zeros((), device=device),
    )
    distributions = None
    with loader.prefetch(device, uses_cuda, replay_prefetch_depth) as prefetched_batches:
        for batch_index, batch in enumerate(prefetched_batches):
            learning_rate = warmup_scaled_learning_rate(
                base_learning_rate,
                warmup_optimizer_steps,
                completed_optimizer_steps + batch_index,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group['lr'] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision is TrainingPrecision.BFLOAT16,
            ):
                output = distributed_model(batch.states)
                loss = objective.calculate_loss(output, batch)
            if collect_distributions and distributions is None:
                distributions = capture_training_distributions(
                    output,
                    batch,
                    objective,
                    source_generation,
                    time.time(),
                )
            loss.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(distributed_model.parameters(), maximum_gradient_norm)
            optimizer.step()
            totals.policy.add_(loss.policy.detach())
            totals.wdl.add_(loss.wdl.detach())
            totals.total.add_(loss.total.detach())
            totals.gradient_norm.add_(gradient_norm.detach())
            if loss.auxiliary:
                totals.auxiliary.add_(torch.stack(tuple(auxiliary.detach() for auxiliary in loss.auxiliary)))
    return _TrainingBatchResult(totals, distributions)


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
    save_path: Path,
) -> CheckpointReference | None:
    if rank != 0:
        return None
    generation = command.target_progress.model_generation
    save_model_and_optimizer(model, optimizer, generation, save_path)
    return CheckpointReference.load(save_path, generation)


def train_rank_quantum(
    rank: int,
    world_size: int,
    configuration: ExperimentConfiguration,
    game: GameImplementation,
    model: Network,
    ddp: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    save_path: Path,
    command: TrainQuantumCommand,
) -> RankTrainingResult:
    started_at = time.perf_counter()
    optimizer_steps = (
        command.target_progress.completed_optimizer_steps - command.source_progress.completed_optimizer_steps
    )
    uses_cuda = configuration.training.topology.trainer.device_type == 'cuda'
    loader = MappedReplayBatchLoader(
        replay=command.replay,
        state=game.state,
        source_optimizer_step=command.replay_source_optimizer_steps,
        optimizer_steps=optimizer_steps,
        global_batch_size=configuration.training.trainer.global_batch_size,
        world_size=world_size,
        rank=rank,
        sampler_seed=configuration.training.random_seed,
        pin_memory=uses_cuda,
    )
    training_result = _train_batches(
        loader,
        ddp,
        optimizer,
        device,
        uses_cuda,
        configuration.training.trainer.max_grad_norm,
        command.parameters.objective,
        collect_distributions=rank == 0,
        source_generation=command.source_progress.model_generation,
        precision=configuration.training.trainer.precision,
        base_learning_rate=command.parameters.learning_rate,
        warmup_optimizer_steps=configuration.training.trainer.warmup_optimizer_steps,
        completed_optimizer_steps=command.source_progress.completed_optimizer_steps,
        replay_prefetch_depth=configuration.training.trainer.replay_prefetch_depth,
    )
    totals = _resolve_loss_totals(training_result.totals)
    checkpoint = _save_rank_checkpoint(rank, model, optimizer, command, save_path)
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
        elapsed_seconds=time.perf_counter() - started_at,
        checkpoint=checkpoint,
        distributions=training_result.distributions,
    )
