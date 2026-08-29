from __future__ import annotations

import math
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
from src.replay.batch_loader import MappedReplayBatchLoader, SearchBudgetLabelledBatches
from src.training.batch import TrainingBatch, TrainingModelOutput
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.paths import checkpoint_manifest_path
from src.training.checkpoint.persistence import load_model_and_optimizer, save_model_and_optimizer
from src.training.configuration import TrainerTopologyParams, TrainingCompilation, TrainingPrecision
from src.training.distributions import (
    AuxiliaryTrainingDistribution,
    TrainingDistributionSnapshot,
    capture_training_distributions,
)
from src.training.network import POLICY_PRIOR_PROBE_POSITIONS, Network
from src.training.objective import (
    ObjectiveLoss,
    ResolvedTrainingObjective,
    auxiliary_batch_weight,
)
from src.training.targets import search_budget_auxiliary_index
from src.training.trainer.contracts import (
    RankTrainingFailure,
    RankTrainingResult,
    SearchBudgetHeadStatistics,
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
    term_trunk_gradients: tuple[float, ...]


@dataclass(frozen=True)
class _DeviceLossTotals:
    policy: torch.Tensor
    wdl: torch.Tensor
    auxiliary: torch.Tensor
    total: torch.Tensor
    gradient_norm: torch.Tensor
    term_trunk_gradients: torch.Tensor
    term_trunk_gradient_probes: torch.Tensor


@dataclass(frozen=True)
class _SearchBudgetHeadTotals:
    auxiliary_index: int
    sums: torch.Tensor

    @staticmethod
    def empty(auxiliary_index: int, device: torch.device) -> _SearchBudgetHeadTotals:
        return _SearchBudgetHeadTotals(auxiliary_index, torch.zeros(8, device=device))

    def accumulate(self, output: TrainingModelOutput, batch: TrainingBatch, loss: ObjectiveLoss) -> None:
        index = self.auxiliary_index
        targets = batch.auxiliary_targets[index].detach().float().squeeze(1)
        predictions = torch.sigmoid(output.auxiliary_logits[index].detach().float()).squeeze(1)
        self.sums.add_(
            torch.stack(
                (
                    loss.auxiliary[index].detach().float(),
                    targets.sum(),
                    (targets * targets).sum(),
                    predictions.sum(),
                    (predictions * predictions).sum(),
                    (predictions - targets).abs().sum(),
                    torch.full((), float(targets.shape[0]), device=targets.device),
                    torch.ones((), device=targets.device),
                )
            )
        )

    def resolve(self, labelled_pool_rows: int) -> SearchBudgetHeadStatistics:
        (
            loss_sum,
            target_sum,
            target_square_sum,
            prediction_sum,
            prediction_square_sum,
            absolute_error_sum,
            row_count,
            batch_count,
        ) = (float(value) for value in self.sums.cpu())
        rows = row_count if row_count > 0.0 else 1.0
        batches = batch_count if batch_count > 0.0 else 1.0
        target_mean = target_sum / rows
        prediction_mean = prediction_sum / rows
        return SearchBudgetHeadStatistics(
            auxiliary_index=self.auxiliary_index,
            labelled_pool_rows=labelled_pool_rows,
            labelled_batches=int(batch_count),
            loss=loss_sum / batches,
            target_mean=target_mean,
            target_standard_deviation=_standard_deviation(target_square_sum, target_mean, rows),
            prediction_mean=prediction_mean,
            prediction_standard_deviation=_standard_deviation(prediction_square_sum, prediction_mean, rows),
            absolute_error_mean=absolute_error_sum / rows,
        )


def _standard_deviation(square_sum: float, mean: float, rows: float) -> float:
    return math.sqrt(max(square_sum / rows - mean * mean, 0.0))


@dataclass(frozen=True)
class _TrainingBatchResult:
    totals: _DeviceLossTotals
    distributions: TrainingDistributionSnapshot | None
    search_budget_head: SearchBudgetHeadStatistics | None


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
    gradient_probe_interval_steps: int,
    search_budget_auxiliary: int | None,
) -> _TrainingBatchResult:
    totals = _DeviceLossTotals(
        policy=torch.zeros((), device=device),
        wdl=torch.zeros((), device=device),
        auxiliary=torch.zeros(len(objective.auxiliary_losses), device=device),
        total=torch.zeros((), device=device),
        gradient_norm=torch.zeros((), device=device),
        term_trunk_gradients=torch.zeros(2 + len(objective.auxiliary_losses), device=device),
        term_trunk_gradient_probes=torch.zeros((), device=device),
    )
    head_totals = (
        None if search_budget_auxiliary is None else _SearchBudgetHeadTotals.empty(search_budget_auxiliary, device)
    )
    distributions = None
    head_distribution = None
    with loader.prefetch(device, uses_cuda, replay_prefetch_depth) as prefetched_batches:
        for batch_index, batch in enumerate(prefetched_batches):
            labelled_batch = loader.is_labelled_batch(batch_index)
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
                loss = objective.calculate_loss(output, batch, search_budget_labelled_batch=labelled_batch)
            if collect_distributions and ((head_distribution is None) if labelled_batch else (distributions is None)):
                snapshot = capture_training_distributions(
                    output,
                    batch,
                    objective,
                    source_generation,
                    time.time(),
                )
                if labelled_batch:
                    assert search_budget_auxiliary is not None
                    head_distribution = snapshot.auxiliary[search_budget_auxiliary]
                else:
                    distributions = snapshot
            if gradient_probe_interval_steps > 0 and batch_index % gradient_probe_interval_steps == 0:
                totals.term_trunk_gradients.add_(
                    _term_trunk_gradients(objective, loss, output.features, labelled_batch)
                )
                totals.term_trunk_gradient_probes.add_(1.0)
            loss.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(distributed_model.parameters(), maximum_gradient_norm)
            optimizer.step()
            totals.policy.add_(loss.policy.detach())
            totals.wdl.add_(loss.wdl.detach())
            totals.total.add_(loss.total.detach())
            totals.gradient_norm.add_(gradient_norm.detach())
            if loss.auxiliary:
                totals.auxiliary.add_(torch.stack(tuple(auxiliary.detach() for auxiliary in loss.auxiliary)))
            if labelled_batch and head_totals is not None:
                head_totals.accumulate(output, batch, loss)
    head_statistics = None if head_totals is None else head_totals.resolve(loader.labelled_pool_rows)
    if head_distribution is not None and distributions is not None:
        assert search_budget_auxiliary is not None
        distributions = _with_search_budget_head_distribution(
            distributions,
            search_budget_auxiliary,
            head_distribution,
        )
    return _TrainingBatchResult(totals, distributions, head_statistics)


def _with_search_budget_head_distribution(
    distributions: TrainingDistributionSnapshot,
    auxiliary_index: int,
    head_distribution: AuxiliaryTrainingDistribution,
) -> TrainingDistributionSnapshot:
    """An ordinary batch holds a handful of labelled rows, so the head's histograms come from a labelled batch."""
    auxiliary = tuple(
        head_distribution if index == auxiliary_index else entry for index, entry in enumerate(distributions.auxiliary)
    )
    return distributions.model_copy(update={'auxiliary': auxiliary})


def _term_trunk_gradients(
    objective: ResolvedTrainingObjective,
    loss: ObjectiveLoss,
    features: torch.Tensor,
    search_budget_labelled_batch: bool,
) -> torch.Tensor:
    """Norm of each weighted term's gradient at the shared trunk, so terms are comparable across heads."""
    weighted = (
        objective.policy_loss_weight * loss.policy,
        objective.value_loss_weight * loss.wdl,
        *(
            auxiliary_batch_weight(configuration, search_budget_labelled_batch) * value
            for configuration, value in zip(objective.auxiliary_losses, loss.auxiliary, strict=True)
        ),
    )
    norms = []
    for term in weighted:
        (gradient,) = torch.autograd.grad(term, features, retain_graph=True, allow_unused=True)
        norms.append(torch.zeros((), device=features.device) if gradient is None else gradient.detach().norm())
    return torch.stack(norms)


def _resolve_loss_totals(totals: _DeviceLossTotals) -> _LossTotals:
    host_totals = torch.cat(
        (
            torch.stack((totals.policy, totals.wdl, totals.total, totals.gradient_norm)),
            totals.auxiliary,
            totals.term_trunk_gradients,
            torch.stack((totals.term_trunk_gradient_probes,)),
        )
    ).cpu()
    term_start = 4 + totals.auxiliary.shape[0]
    term_end = term_start + totals.term_trunk_gradients.shape[0]
    probes = float(host_totals[term_end])
    divisor = probes if probes > 0.0 else 1.0
    return _LossTotals(
        policy=float(host_totals[0]),
        wdl=float(host_totals[1]),
        auxiliary=tuple(float(value) for value in host_totals[4:term_start]),
        total=float(host_totals[2]),
        gradient_norm=float(host_totals[3]),
        term_trunk_gradients=tuple(float(value) / divisor for value in host_totals[term_start:term_end]),
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
    labelled_batches = _labelled_batch_plan(configuration, command)
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
        labelled_batches=labelled_batches,
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
        gradient_probe_interval_steps=configuration.training.trainer.gradient_probe_interval_steps,
        search_budget_auxiliary=None if labelled_batches is None else labelled_batches.auxiliary_index,
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
        auxiliary_losses=_auxiliary_losses_with_head_batches(
            tuple(value / divisor for value in totals.auxiliary),
            training_result.search_budget_head,
        ),
        total_loss=totals.total / divisor,
        gradient_norm=totals.gradient_norm / divisor,
        term_trunk_gradients=totals.term_trunk_gradients,
        elapsed_seconds=time.perf_counter() - started_at,
        checkpoint=checkpoint,
        distributions=training_result.distributions,
        search_budget_head=training_result.search_budget_head,
    )


def _labelled_batch_plan(
    configuration: ExperimentConfiguration,
    command: TrainQuantumCommand,
) -> SearchBudgetLabelledBatches | None:
    head_training = configuration.training.lifecycle.search_budget.head_training
    if not head_training.dedicated_batches:
        return None
    auxiliary_index = search_budget_auxiliary_index(command.replay.layout.targets.auxiliary_heads)
    if auxiliary_index is None:
        return None
    return SearchBudgetLabelledBatches(
        auxiliary_index=auxiliary_index,
        interval_optimizer_steps=head_training.interval_optimizer_steps,
    )


def _auxiliary_losses_with_head_batches(
    losses: tuple[float, ...],
    head_statistics: SearchBudgetHeadStatistics | None,
) -> tuple[float, ...]:
    if head_statistics is None:
        return losses
    return tuple(
        head_statistics.loss if index == head_statistics.auxiliary_index else value
        for index, value in enumerate(losses)
    )
