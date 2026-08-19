from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time

import numpy as np
import torch
from pydantic import Field
from src.experiment.configuration import load_experiment_configuration
from src.games.composition import create_game_implementation
from src.games.representation import NetworkDimensions
from src.replay.batch_loader import build_training_batch
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.training.batch import TrainingBatch
from src.training.checkpoint.persistence import create_model, create_optimizer
from src.training.configuration import TrainingPrecision
from src.training.network import (
    AttentionNetworkParams,
    Chess76PlaneDirectPolicyHeadConfiguration,
    GlobalPoolingResidualContext,
    NetworkConfiguration,
    NetworkDefinition,
    Network,
    NetworkParams,
    ResidualContextPlacement,
)
from src.training.objective import (
    ObjectiveLoss,
    ResolvedAuxiliaryLoss,
    ResolvedLegalMovesLoss,
    ResolvedNextPolicyLoss,
    ResolvedTrainingObjective,
    blended_wdl_targets,
)
from src.training.targets import AuxiliaryHeadLayout
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


class BenchmarkModel(str, Enum):
    ATTENTION_500K = 'attention-500k'
    ATTENTION_1M = 'attention-1m'
    CNN_1M = 'cnn-1m'


class ObjectiveProfile(str, Enum):
    PRODUCTION = 'production'
    HALF_AUXILIARY = 'half-auxiliary'
    PRIMARY_ONLY = 'primary-only'
    WITHOUT_LEGAL_MOVES = 'without-legal-moves'


@dataclass(frozen=True)
class Arguments:
    resolved_configuration: Path
    run_directory: Path
    device_id: int
    batch_size: int
    maximum_optimizer_steps: int
    report_interval: int
    model_generation: int
    random_seed: int
    models: tuple[BenchmarkModel, ...]
    objective_profile: ObjectiveProfile
    output_path: Path


class LossValues(FrozenModel):
    policy: float
    wdl: float
    auxiliary: tuple[float, ...]
    total: float


class LossObservation(FrozenModel):
    optimizer_step: int
    elapsed_seconds: float
    training: LossValues
    holdout: LossValues
    training_excess_ratio_to_floor: float


class OverfitBenchmarkResult(FrozenModel):
    model_id: BenchmarkModel
    network: NetworkDefinition
    objective_profile: ObjectiveProfile
    model_generation: int
    parameter_count: int
    trainable_parameter_count: int
    batch_size: int
    learning_rate: float
    precision: TrainingPrecision
    maximum_optimizer_steps: int
    completed_optimizer_steps: int
    elapsed_seconds: float
    optimizer_steps_per_second: float
    peak_allocated_mib: float
    achievable_loss_floor: LossValues
    steps_to_fifty_percent_above_floor: int | None
    steps_to_ten_percent_above_floor: int | None
    steps_to_one_percent_above_floor: int | None
    observations: tuple[LossObservation, ...]


class OverfitBenchmarkReport(FrozenModel):
    schema_version: int = Field(default=1)
    replay_path: Path
    replay_rows: int
    batch_size: int
    model_generation: int
    random_seed: int
    objective_profile: ObjectiveProfile
    results: tuple[OverfitBenchmarkResult, ...]


@dataclass(frozen=True)
class _BenchmarkInputs:
    dimensions: NetworkDimensions
    auxiliary_heads: tuple[AuxiliaryHeadLayout, ...]
    training_batch: TrainingBatch
    holdout_batch: TrainingBatch
    objective: ResolvedTrainingObjective
    learning_rate: float
    precision: TrainingPrecision
    replay_rows: int


def benchmark_network(model_id: BenchmarkModel) -> NetworkConfiguration:
    policy_head = Chess76PlaneDirectPolicyHeadConfiguration()
    match model_id:
        case BenchmarkModel.ATTENTION_500K:
            return AttentionNetworkParams(
                num_layers=6,
                embedding_size=96,
                num_heads=3,
                feedforward_size=192,
                dropout=0.0,
                policy_head=policy_head,
                num_value_channels=2,
                value_fc_size=48,
            )
        case BenchmarkModel.ATTENTION_1M:
            return AttentionNetworkParams(
                num_layers=8,
                embedding_size=128,
                num_heads=4,
                feedforward_size=256,
                dropout=0.0,
                policy_head=policy_head,
                num_value_channels=2,
                value_fc_size=48,
            )
        case BenchmarkModel.CNN_1M:
            return NetworkParams(
                num_layers=8,
                hidden_size=88,
                residual_context=GlobalPoolingResidualContext(
                    placement=ResidualContextPlacement.EVERY_SECOND_BLOCK,
                ),
                policy_head=policy_head,
                num_value_channels=2,
                value_fc_size=48,
            )


def objective_for_profile(
    objective: ResolvedTrainingObjective,
    profile: ObjectiveProfile,
) -> ResolvedTrainingObjective:
    match profile:
        case ObjectiveProfile.PRODUCTION:
            return objective
        case ObjectiveProfile.HALF_AUXILIARY:
            auxiliary_scale = 0.5
        case ObjectiveProfile.PRIMARY_ONLY:
            auxiliary_scale = 0.0
        case ObjectiveProfile.WITHOUT_LEGAL_MOVES:
            auxiliary_scale = 1.0
    auxiliary_losses: list[ResolvedAuxiliaryLoss] = []
    for loss in objective.auxiliary_losses:
        match loss:
            case ResolvedLegalMovesLoss() if profile is ObjectiveProfile.WITHOUT_LEGAL_MOVES:
                weight = 0.0
            case _:
                weight = loss.weight * auxiliary_scale
        auxiliary_losses.append(loss.validated_copy(update={'weight': weight}))
    return objective.validated_copy(
        update={'auxiliary_losses': [loss.model_dump(mode='json') for loss in auxiliary_losses]}
    )


def achievable_loss_floor(
    batch: TrainingBatch,
    objective: ResolvedTrainingObjective,
) -> LossValues:
    sample_weights = batch.sample_weights / batch.sample_weights.mean()
    policy = _weighted_entropy(batch.policy_targets, sample_weights)
    wdl_targets = blended_wdl_targets(batch.wdl_targets, batch.root_values, objective.root_value_blend)
    wdl = _weighted_entropy(wdl_targets, sample_weights)
    auxiliary = tuple(
        _auxiliary_loss_floor(loss, target, eligibility, sample_weights)
        for loss, target, eligibility in zip(
            objective.auxiliary_losses,
            batch.auxiliary_targets,
            batch.auxiliary_eligibility,
        )
    )
    total = objective.policy_loss_weight * policy + objective.value_loss_weight * wdl
    total += sum(loss.weight * floor for loss, floor in zip(objective.auxiliary_losses, auxiliary))
    return LossValues(policy=policy, wdl=wdl, auxiliary=auxiliary, total=total)


def _weighted_entropy(targets: torch.Tensor, sample_weights: torch.Tensor) -> float:
    rows = -(targets * targets.clamp_min(torch.finfo(targets.dtype).tiny).log()).sum(dim=1)
    return float((rows * sample_weights).mean())


def _auxiliary_loss_floor(
    loss: ResolvedAuxiliaryLoss,
    targets: torch.Tensor,
    eligibility: torch.Tensor,
    sample_weights: torch.Tensor,
) -> float:
    match loss:
        case ResolvedNextPolicyLoss():
            eligible_weights = eligibility.to(dtype=sample_weights.dtype) * sample_weights
            denominator = eligible_weights.sum().clamp_min(1.0)
            rows = -(targets * targets.clamp_min(torch.finfo(targets.dtype).tiny).log()).sum(dim=1)
            return float((rows * eligible_weights).sum() / denominator)
        case _:
            return 0.0


def _loss_values(loss: ObjectiveLoss) -> LossValues:
    return LossValues(
        policy=float(loss.policy.detach()),
        wdl=float(loss.wdl.detach()),
        auxiliary=tuple(float(value.detach()) for value in loss.auxiliary),
        total=float(loss.total.detach()),
    )


def _evaluate_loss(
    model: Network,
    batch: TrainingBatch,
    objective: ResolvedTrainingObjective,
    precision: TrainingPrecision,
) -> LossValues:
    model.eval()
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=batch.states.device.type,
            dtype=torch.bfloat16,
            enabled=precision is TrainingPrecision.BFLOAT16,
        ),
    ):
        loss = objective.calculate_loss(model.training_output(batch.states), batch)
    model.train()
    return _loss_values(loss)


def _training_excess_ratio_to_floor(loss: LossValues, floor: LossValues) -> float:
    return max(0.0, loss.total - floor.total) / floor.total


def run_model_benchmark(
    arguments: Arguments,
    model_id: BenchmarkModel,
    inputs: _BenchmarkInputs,
) -> OverfitBenchmarkResult:
    torch.manual_seed(arguments.random_seed)
    torch.cuda.manual_seed_all(arguments.random_seed)
    training_batch = inputs.training_batch
    holdout_batch = inputs.holdout_batch
    objective = inputs.objective
    device = training_batch.states.device
    network = benchmark_network(model_id)
    model = create_model(network, device, inputs.dimensions, inputs.auxiliary_heads)
    optimizer = create_optimizer(model, 'adamw')
    for parameter_group in optimizer.param_groups:
        parameter_group['lr'] = inputs.learning_rate
    floor = achievable_loss_floor(training_batch, objective)
    initial_training = _evaluate_loss(model, training_batch, objective, inputs.precision)
    initial_holdout = _evaluate_loss(model, holdout_batch, objective, inputs.precision)
    observations: list[LossObservation] = [
        LossObservation(
            optimizer_step=0,
            elapsed_seconds=0.0,
            training=initial_training,
            holdout=initial_holdout,
            training_excess_ratio_to_floor=_training_excess_ratio_to_floor(initial_training, floor),
        )
    ]
    steps_to_fifty_percent: int | None = None
    steps_to_ten_percent: int | None = None
    steps_to_one_percent: int | None = None
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    completed_steps = 0
    for optimizer_step in range(1, arguments.maximum_optimizer_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=inputs.precision is TrainingPrecision.BFLOAT16,
        ):
            loss = objective.calculate_loss(model.training_output(training_batch.states), training_batch)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        completed_steps = optimizer_step
        if optimizer_step % arguments.report_interval and optimizer_step != arguments.maximum_optimizer_steps:
            continue
        torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - started_at
        training_loss = _evaluate_loss(model, training_batch, objective, inputs.precision)
        holdout_loss = _evaluate_loss(model, holdout_batch, objective, inputs.precision)
        excess_ratio = _training_excess_ratio_to_floor(training_loss, floor)
        observations.append(
            LossObservation(
                optimizer_step=optimizer_step,
                elapsed_seconds=elapsed_seconds,
                training=training_loss,
                holdout=holdout_loss,
                training_excess_ratio_to_floor=excess_ratio,
            )
        )
        if steps_to_fifty_percent is None and excess_ratio <= 0.5:
            steps_to_fifty_percent = optimizer_step
        if steps_to_ten_percent is None and excess_ratio <= 0.1:
            steps_to_ten_percent = optimizer_step
        if steps_to_one_percent is None and excess_ratio <= 0.01:
            steps_to_one_percent = optimizer_step
            break
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at
    result = OverfitBenchmarkResult(
        model_id=model_id,
        network=NetworkDefinition(
            architecture=network,
            dimensions=inputs.dimensions,
            auxiliary_heads=inputs.auxiliary_heads,
        ),
        objective_profile=arguments.objective_profile,
        model_generation=arguments.model_generation,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameter_count=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        batch_size=len(training_batch),
        learning_rate=inputs.learning_rate,
        precision=inputs.precision,
        maximum_optimizer_steps=arguments.maximum_optimizer_steps,
        completed_optimizer_steps=completed_steps,
        elapsed_seconds=elapsed_seconds,
        optimizer_steps_per_second=completed_steps / elapsed_seconds,
        peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
        achievable_loss_floor=floor,
        steps_to_fifty_percent_above_floor=steps_to_fifty_percent,
        steps_to_ten_percent_above_floor=steps_to_ten_percent,
        steps_to_one_percent_above_floor=steps_to_one_percent,
        observations=tuple(observations),
    )
    del optimizer
    del model
    torch.cuda.empty_cache()
    return result


def _load_fixed_batches(
    arguments: Arguments,
) -> _BenchmarkInputs:
    configuration = load_experiment_configuration(arguments.resolved_configuration)
    game = create_game_implementation(configuration)
    replay_layout = ReplayLayout(
        packed_planes=game.state.packed_plane_layout,
        targets=game.target_layout,
        maximum_policy_entries=configuration.training.lifecycle.replay.maximum_policy_entries,
        maximum_legal_actions=game.state.maximum_legal_action_count,
    )
    dimensions = game.network_dimensions
    auxiliary_heads = game.target_layout.auxiliary_heads
    objective = objective_for_profile(
        game.training_objective_at(arguments.model_generation),
        arguments.objective_profile,
    )
    replay_store = ReplayStore.open(arguments.run_directory / 'replay.bin', replay_layout, writable=False)
    try:
        replay_rows = replay_store.state.size
        required_rows = arguments.batch_size * 2
        if replay_rows < required_rows:
            raise ValueError(f'Replay requires at least {required_rows} rows for disjoint train and holdout batches.')
        generator = np.random.default_rng(arguments.random_seed)
        sample_indices = generator.choice(replay_rows, size=required_rows, replace=False)
        augmentation_indices = generator.integers(0, game.state.augmentation_count, size=required_rows)
        training_batch = build_training_batch(
            replay_store,
            game.state,
            tuple(int(index) for index in sample_indices[: arguments.batch_size]),
            tuple(int(index) for index in augmentation_indices[: arguments.batch_size]),
        )
        holdout_batch = build_training_batch(
            replay_store,
            game.state,
            tuple(int(index) for index in sample_indices[arguments.batch_size :]),
            tuple(int(index) for index in augmentation_indices[arguments.batch_size :]),
        )
    finally:
        replay_store.close()
        game.close()
    device = torch.device('cuda', arguments.device_id)
    training_batch = training_batch.to_device(device, non_blocking=False)
    holdout_batch = holdout_batch.to_device(device, non_blocking=False)
    learning_rate = configuration.training.trainer.learning_rate.value_at(arguments.model_generation)
    return _BenchmarkInputs(
        dimensions=dimensions,
        auxiliary_heads=auxiliary_heads,
        training_batch=training_batch,
        holdout_batch=holdout_batch,
        objective=objective,
        learning_rate=learning_rate,
        precision=configuration.training.trainer.precision,
        replay_rows=replay_rows,
    )


def run_benchmark(arguments: Arguments) -> OverfitBenchmarkReport:
    torch.cuda.set_device(arguments.device_id)
    inputs = _load_fixed_batches(arguments)
    results = tuple(
        run_model_benchmark(
            arguments,
            model_id,
            inputs,
        )
        for model_id in arguments.models
    )
    report = OverfitBenchmarkReport(
        replay_path=arguments.run_directory / 'replay.bin',
        replay_rows=inputs.replay_rows,
        batch_size=arguments.batch_size,
        model_generation=arguments.model_generation,
        random_seed=arguments.random_seed,
        objective_profile=arguments.objective_profile,
        results=results,
    )
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomically(arguments.output_path, report.model_dump_json(indent=2) + '\n')
    return report


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Compare how quickly Chess networks overfit one fixed replay batch.')
    parser.add_argument('--resolved-configuration', required=True, type=Path)
    parser.add_argument('--run-directory', required=True, type=Path)
    parser.add_argument('--device-id', default=0, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--maximum-optimizer-steps', default=2000, type=int)
    parser.add_argument('--report-interval', default=25, type=int)
    parser.add_argument('--model-generation', default=94, type=int)
    parser.add_argument('--random-seed', default=20260819, type=int)
    parser.add_argument(
        '--models',
        choices=tuple(model.value for model in BenchmarkModel),
        default=tuple(model.value for model in BenchmarkModel),
        nargs='+',
        type=BenchmarkModel,
    )
    parser.add_argument(
        '--objective-profile',
        choices=tuple(profile.value for profile in ObjectiveProfile),
        default=ObjectiveProfile.PRODUCTION.value,
        type=ObjectiveProfile,
    )
    parser.add_argument('--output-path', required=True, type=Path)
    namespace = parser.parse_args()
    arguments = Arguments(
        resolved_configuration=namespace.resolved_configuration,
        run_directory=namespace.run_directory,
        device_id=namespace.device_id,
        batch_size=namespace.batch_size,
        maximum_optimizer_steps=namespace.maximum_optimizer_steps,
        report_interval=namespace.report_interval,
        model_generation=namespace.model_generation,
        random_seed=namespace.random_seed,
        models=tuple(namespace.models),
        objective_profile=namespace.objective_profile,
        output_path=namespace.output_path,
    )
    if not arguments.resolved_configuration.is_file():
        raise ValueError(f'Resolved configuration does not exist: {arguments.resolved_configuration}')
    if not arguments.run_directory.is_dir():
        raise ValueError(f'Run directory does not exist: {arguments.run_directory}')
    if (
        min(
            arguments.batch_size,
            arguments.maximum_optimizer_steps,
            arguments.report_interval,
        )
        <= 0
    ):
        raise ValueError('Batch size, optimizer steps, and report interval must be positive.')
    if arguments.device_id < 0 or arguments.model_generation < 0:
        raise ValueError('Device ID and model generation must be nonnegative.')
    if arguments.maximum_optimizer_steps % arguments.report_interval:
        raise ValueError('Maximum optimizer steps must be divisible by the report interval.')
    return arguments


def main() -> None:
    print(run_benchmark(parse_arguments()).model_dump_json())


if __name__ == '__main__':
    main()
