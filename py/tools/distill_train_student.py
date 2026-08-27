from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from src.distillation.dataset import build_training_batch, open_dataset, read_manifest
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS, CHESS_STATE_CONTRACT
from src.training.batch import TrainingBatch
from src.training.checkpoint.contracts import CheckpointManifest
from src.training.checkpoint.paths import checkpoint_manifest_path, model_save_path, optimizer_save_path
from src.training.checkpoint.persistence import create_model
from src.training.model_cost import format_model_cost, measure_model_cost
from src.training.network import (
    AttentionNetworkParams,
    ChessFromToAttentionPolicyHeadConfiguration,
    DensePolicyHeadConfiguration,
    DisabledAttentionBiasConfiguration,
    GlobalPoolingResidualContext,
    InferenceNetwork,
    Network,
    NetworkConfiguration,
    NetworkParams,
    PolicyHeadConfiguration,
    RelativeAttentionBiasConfiguration,
    ResidualContextPlacement,
    SmolgenAttentionBiasConfiguration,
)
from src.training.objective import ObjectiveLoss, ResolvedTrainingObjective, resolve_auxiliary_losses
from src.training.targets import (
    AuxiliaryHeadLayout,
    AuxiliaryTargetConfiguration,
    NextPolicyTargetConfiguration,
    RemainingGameLengthTargetConfiguration,
    build_training_target_layout,
)
from src.util.atomic_file import write_text_atomically
from src.util.generation_schedule import ConstantSchedule
from src.util.hashing import file_sha256
from src.util.log import log
from tools.benchmark_training_overfit import LossValues, achievable_loss_floor

HELD_OUT_EVALUATION_BATCHES = 8
DISTILLABLE_AUXILIARY_HEADS = ('next_policy', 'remaining_game_length')
AUXILIARY_LOSS_WEIGHT = 0.1
NEXT_POLICY_PLY_OFFSET = 1
REMAINING_GAME_LENGTH_NORMALIZATION_SCALE = 400.0


class LearningRateSchedule(str, Enum):
    COSINE = 'cosine'
    PLATEAU = 'plateau'
    # The near-flat production shape, the proposed staged decay and a cosine that stops at the same floor.
    PRODUCTION_FLAT = 'production_flat'
    STAGED_DECAY = 'staged_decay'
    COSINE_FLOOR = 'cosine_floor'


class OptimizerKind(str, Enum):
    ADAMW = 'adamw'
    SGD_MOMENTUM = 'sgd_momentum'


class NetworkKind(str, Enum):
    CONVOLUTIONAL = 'convolutional'
    ATTENTION = 'attention'


class PolicyHeadKind(str, Enum):
    DENSE = 'dense'
    FROM_TO_ATTENTION = 'from_to_attention'


class AttentionBiasKind(str, Enum):
    DISABLED = 'disabled'
    RELATIVE = 'relative'
    SMOLGEN = 'smolgen'


# Self-play schedules are authored against generations, so they are mapped onto a supervised step budget as
# (fraction of the run, multiplier on the peak). The reference horizon is generation 1200 at 500 steps each,
# which is the last generation the staged proposal drops at plus room to sit at the floor.
SELF_PLAY_REFERENCE_GENERATIONS = 1200
# 0.005 -> 0.004 at generation 100 -> 0.003 at generation 1000, the schedule production v9 runs.
PRODUCTION_FLAT_STAGES = ((0.0, 1.0), (100 / 1200, 0.8), (1000 / 1200, 0.6))
# The proposed v10 drops at generations 200 / 600 / 1000 from 0.005 to 0.0005; the two intermediate rates
# are not pinned by the proposal, so they are placed at equal ratios along the way to the floor.
STAGED_DECAY_STAGES = ((0.0, 1.0), (200 / 1200, 0.1 ** (1 / 3)), (600 / 1200, 0.1 ** (2 / 3)), (1000 / 1200, 0.1))
# AlphaZero: 0.2 -> 0.02 -> 0.002 -> 0.0002 at 0 / 100k / 300k / 500k steps of 700k.
ALPHAZERO_SGD_STAGES = ((0.0, 1.0), (1.0 / 7.0, 0.1), (3.0 / 7.0, 0.01), (5.0 / 7.0, 0.001))


@dataclass(frozen=True)
class Arguments:
    dataset: Path
    output_run_state: Path
    network_kind: NetworkKind
    layers: int
    hidden_size: int
    heads: int
    feedforward: int
    policy_head_kind: PolicyHeadKind
    policy_key_size: int
    attention_bias_kind: AttentionBiasKind
    smolgen_compressed_size: int
    smolgen_hidden_size: int
    smolgen_generated_size: int
    optimizer_kind: OptimizerKind
    floor_fraction: float
    policy_bottleneck_rank: int
    batch_size: int
    steps: int
    learning_rate: float
    learning_rate_schedule: LearningRateSchedule
    anneal_fraction: float
    warmup_steps: int
    max_grad_norm: float
    holdout_fraction: float
    training_fraction: float
    evaluate_every: int
    checkpoint_every: int
    distil_auxiliary_heads: tuple[str, ...]
    device_id: int
    random_seed: int
    generation: int


@dataclass(frozen=True)
class DatasetSplit:
    training_row_count: int
    held_out_start_row: int
    held_out_row_count: int


@dataclass(frozen=True)
class ParameterCounts:
    backbone: int
    heads: int
    auxiliary_heads: int

    @property
    def total(self) -> int:
        return self.backbone + self.heads + self.auxiliary_heads


def dataset_split(row_count: int, holdout_fraction: float, training_fraction: float) -> DatasetSplit:
    held_out_row_count = max(1, round(row_count * holdout_fraction))
    held_out_start_row = row_count - held_out_row_count
    return DatasetSplit(
        training_row_count=max(1, round(held_out_start_row * training_fraction)),
        held_out_start_row=held_out_start_row,
        held_out_row_count=held_out_row_count,
    )


def student_policy_head(arguments: Arguments) -> PolicyHeadConfiguration:
    match arguments.policy_head_kind:
        case PolicyHeadKind.DENSE:
            return DensePolicyHeadConfiguration(channels=4, bottleneck_rank=arguments.policy_bottleneck_rank or None)
        case PolicyHeadKind.FROM_TO_ATTENTION:
            return ChessFromToAttentionPolicyHeadConfiguration(key_size=arguments.policy_key_size)


def student_attention_bias(arguments: Arguments):
    match arguments.attention_bias_kind:
        case AttentionBiasKind.DISABLED:
            return DisabledAttentionBiasConfiguration()
        case AttentionBiasKind.RELATIVE:
            return RelativeAttentionBiasConfiguration()
        case AttentionBiasKind.SMOLGEN:
            return SmolgenAttentionBiasConfiguration(
                compressed_size=arguments.smolgen_compressed_size,
                hidden_size=arguments.smolgen_hidden_size,
                generated_size=arguments.smolgen_generated_size,
            )


def student_architecture(arguments: Arguments) -> NetworkConfiguration:
    match arguments.network_kind:
        case NetworkKind.CONVOLUTIONAL:
            return NetworkParams(
                num_layers=arguments.layers,
                hidden_size=arguments.hidden_size,
                residual_context=GlobalPoolingResidualContext(placement=ResidualContextPlacement.EVERY_SECOND_BLOCK),
                policy_head=student_policy_head(arguments),
                num_value_channels=2,
                value_fc_size=48,
            )
        case NetworkKind.ATTENTION:
            return AttentionNetworkParams(
                num_layers=arguments.layers,
                embedding_size=arguments.hidden_size,
                num_heads=arguments.heads,
                feedforward_size=arguments.feedforward,
                dropout=0.0,
                attention_bias=student_attention_bias(arguments),
                policy_head=student_policy_head(arguments),
                num_value_channels=2,
                value_fc_size=48,
            )


def create_student_optimizer(model: Network, kind: OptimizerKind, peak_learning_rate: float) -> torch.optim.Optimizer:
    match kind:
        case OptimizerKind.ADAMW:
            return torch.optim.AdamW(
                model.parameters(), lr=peak_learning_rate, weight_decay=0.0001, amsgrad=True, eps=1e-5
            )
        case OptimizerKind.SGD_MOMENTUM:
            return torch.optim.SGD(
                model.parameters(), lr=peak_learning_rate, momentum=0.9, weight_decay=0.0001, nesterov=True
            )


def auxiliary_target_configurations(heads: tuple[str, ...]) -> tuple[AuxiliaryTargetConfiguration, ...]:
    loss_weight = ConstantSchedule[float](value=AUXILIARY_LOSS_WEIGHT)
    configurations: list[AuxiliaryTargetConfiguration] = []
    for head in heads:
        match head:
            case 'next_policy':
                configurations.append(
                    NextPolicyTargetConfiguration(ply_offset=NEXT_POLICY_PLY_OFFSET, loss_weight=loss_weight)
                )
            case 'remaining_game_length':
                configurations.append(
                    RemainingGameLengthTargetConfiguration(
                        loss_weight=loss_weight,
                        normalization_scale=REMAINING_GAME_LENGTH_NORMALIZATION_SCALE,
                    )
                )
            case _:
                raise ValueError(f'The student cannot distil an auxiliary head named {head!r}.')
    return tuple(configurations)


def auxiliary_head_layouts(heads: tuple[str, ...], action_size: int) -> tuple[AuxiliaryHeadLayout, ...]:
    return build_training_target_layout(action_size, auxiliary_target_configurations(heads)).auxiliary_heads


def distillation_objective(auxiliary_heads: tuple[str, ...] = ()) -> ResolvedTrainingObjective:
    return ResolvedTrainingObjective(
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        root_value_blend=0.0,
        auxiliary_losses=resolve_auxiliary_losses(auxiliary_target_configurations(auxiliary_heads), 0),
    )


def parameter_counts(model: Network) -> ParameterCounts:
    def count(*modules: torch.nn.Module) -> int:
        return sum(parameter.numel() for module in modules for parameter in module.parameters())

    return ParameterCounts(
        backbone=count(model.start_block, model.backbone, model.finish_block),
        heads=count(model.policy_head, model.value_head),
        auxiliary_heads=count(model.auxiliary_head_modules),
    )


def staged_multiplier(stages: tuple[tuple[float, float], ...], progress: float) -> float:
    multiplier = stages[0][1]
    for stage_progress, stage_multiplier in stages:
        if progress >= stage_progress:
            multiplier = stage_multiplier
    return multiplier


def learning_rate_at(
    step: int,
    total_steps: int,
    peak_learning_rate: float,
    warmup_steps: int,
    schedule: LearningRateSchedule = LearningRateSchedule.COSINE,
    anneal_fraction: float = 0.2,
    optimizer_kind: OptimizerKind = OptimizerKind.ADAMW,
    floor_fraction: float = 0.1,
) -> float:
    if step <= warmup_steps:
        return peak_learning_rate * step / warmup_steps
    progress = step / total_steps
    match schedule:
        case LearningRateSchedule.PRODUCTION_FLAT:
            return peak_learning_rate * staged_multiplier(PRODUCTION_FLAT_STAGES, progress)
        case LearningRateSchedule.STAGED_DECAY:
            stages = ALPHAZERO_SGD_STAGES if optimizer_kind is OptimizerKind.SGD_MOMENTUM else STAGED_DECAY_STAGES
            return peak_learning_rate * staged_multiplier(stages, progress)
        case LearningRateSchedule.COSINE_FLOOR:
            shape = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return peak_learning_rate * (floor_fraction + (1.0 - floor_fraction) * shape)
        case LearningRateSchedule.COSINE:
            anneal_start = warmup_steps
        case LearningRateSchedule.PLATEAU:
            anneal_start = total_steps - max(1, round(total_steps * anneal_fraction))
    if step <= anneal_start:
        return peak_learning_rate
    decay_progress = (step - anneal_start) / max(total_steps - anneal_start, 1)
    return peak_learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(decay_progress, 1.0)))


def observed_losses(loss: ObjectiveLoss) -> LossValues:
    return LossValues(
        policy=float(loss.policy.detach()),
        wdl=float(loss.wdl.detach()),
        auxiliary=tuple(float(value.detach()) for value in loss.auxiliary),
        total=float(loss.total.detach()),
    )


def mean_loss_values(values: tuple[LossValues, ...]) -> LossValues:
    return LossValues(
        policy=sum(value.policy for value in values) / len(values),
        wdl=sum(value.wdl for value in values) / len(values),
        auxiliary=tuple(sum(head) / len(values) for head in zip(*(value.auxiliary for value in values))),
        total=sum(value.total for value in values) / len(values),
    )


def sample_training_batch(
    records: npt.NDArray,
    training_row_count: int,
    batch_size: int,
    action_size: int,
    generator: np.random.Generator,
    device: torch.device,
    auxiliary_heads: tuple[str, ...] = (),
) -> TrainingBatch:
    indices = np.sort(generator.integers(0, training_row_count, size=batch_size))
    return build_training_batch(records[indices], CHESS_STATE_CONTRACT, action_size, device, auxiliary_heads)


def held_out_batches(
    records: npt.NDArray,
    held_out_start_row: int,
    batch_size: int,
    action_size: int,
    device: torch.device,
    auxiliary_heads: tuple[str, ...] = (),
) -> tuple[TrainingBatch, ...]:
    available_rows = len(records) - held_out_start_row
    rows_per_batch = min(batch_size, available_rows)
    batch_count = min(HELD_OUT_EVALUATION_BATCHES, available_rows // rows_per_batch)
    return tuple(
        build_training_batch(
            records[held_out_start_row + index * rows_per_batch : held_out_start_row + (index + 1) * rows_per_batch],
            CHESS_STATE_CONTRACT,
            action_size,
            device,
            auxiliary_heads,
        )
        for index in range(batch_count)
    )


def evaluate(
    model: Network,
    batches: tuple[TrainingBatch, ...],
    objective: ResolvedTrainingObjective,
    device: torch.device,
) -> LossValues:
    model.eval()
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'),
    ):
        values = tuple(
            observed_losses(objective.calculate_loss(model.training_output(batch.states), batch)) for batch in batches
        )
    model.train()
    return mean_loss_values(values)


def auxiliary_loss_report(heads: tuple[str, ...], values: LossValues) -> str:
    return ''.join(f' {head} {value:.4f}' for head, value in zip(heads, values.auxiliary))


def intermediate_run_state(output_run_state: Path, step: int) -> Path:
    return output_run_state / f'step_{step}'


def save_student_checkpoint(
    model: Network,
    optimizer: torch.optim.Optimizer,
    generation: int,
    save_folder: Path,
) -> CheckpointManifest:
    raw_model_path = model_save_path(generation, save_folder)
    raw_optimizer_path = optimizer_save_path(generation, save_folder)
    jit_model_path = raw_model_path.with_suffix('.jit.pt')

    torch.save(model.state_dict(), raw_model_path)
    torch.save(optimizer.state_dict(), raw_optimizer_path)

    # save_model_and_optimizer rescales the policy head at generation 0 to hit a bootstrap prior shape,
    # which would undo the distilled policy, so the student exports its own inference model.
    inference_model = InferenceNetwork(model)
    inference_model.eval()
    inference_model.fuse_model()
    torch.jit.save(
        torch.jit.script(inference_model),
        str(jit_model_path),
        _extra_files={'network.json': inference_model.checkpoint_definition().model_dump_json()},
    )

    manifest = CheckpointManifest(
        generation=generation,
        network=model.checkpoint_definition(),
        model_path=raw_model_path.name,
        model_sha256=file_sha256(raw_model_path),
        optimizer_path=raw_optimizer_path.name,
        optimizer_sha256=file_sha256(raw_optimizer_path),
        inference_model_path=jit_model_path.name,
        inference_model_sha256=file_sha256(jit_model_path),
    )
    write_text_atomically(checkpoint_manifest_path(generation, save_folder), manifest.model_dump_json(indent=2) + '\n')
    return manifest


def select_device(device_id: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda', device_id)
    return torch.device('cpu')


def train_student(arguments: Arguments) -> CheckpointManifest:
    records, dataset_manifest = open_dataset(arguments.dataset)
    if dataset_manifest.game != CHESS_STATE_CONTRACT.name:
        raise ValueError(f'This student trains on chess datasets, not on {dataset_manifest.game}.')
    if dataset_manifest.action_size != CHESS_NETWORK_DIMENSIONS.actions:
        raise ValueError(f'Dataset action size {dataset_manifest.action_size} does not match the chess action space.')

    split = dataset_split(len(records), arguments.holdout_fraction, arguments.training_fraction)
    if split.training_row_count < arguments.batch_size:
        raise ValueError(
            f'{split.training_row_count} training rows are fewer than one batch of {arguments.batch_size}.'
        )

    device = select_device(arguments.device_id)
    torch.manual_seed(arguments.random_seed)
    torch.cuda.manual_seed_all(arguments.random_seed)

    auxiliary_heads = arguments.distil_auxiliary_heads
    architecture = student_architecture(arguments)
    model = create_model(
        architecture,
        device,
        CHESS_NETWORK_DIMENSIONS,
        auxiliary_head_layouts(auxiliary_heads, dataset_manifest.action_size),
    )
    optimizer = create_student_optimizer(model, arguments.optimizer_kind, arguments.learning_rate)
    objective = distillation_objective(auxiliary_heads)
    counts = parameter_counts(model)
    log(
        f'Student parameters: {counts.total} total = {counts.backbone} backbone + {counts.heads} primary heads + '
        f'{counts.auxiliary_heads} auxiliary heads {auxiliary_heads if auxiliary_heads else "(none)"}.'
    )
    log(format_model_cost(f'Student {arguments.network_kind.value}', measure_model_cost(model)))
    log(f'Student architecture: {architecture.model_dump_json()}')
    log(f'Optimizer {arguments.optimizer_kind.value}, schedule {arguments.learning_rate_schedule.value}.')
    log(
        f'Training on {split.training_row_count} rows, training fraction {arguments.training_fraction:g} of the '
        f'{split.held_out_start_row} rows before the holdout, holding out the last {split.held_out_row_count} rows.'
    )

    evaluation_batches = held_out_batches(
        records,
        split.held_out_start_row,
        arguments.batch_size,
        dataset_manifest.action_size,
        device,
        auxiliary_heads,
    )
    floor = mean_loss_values(tuple(achievable_loss_floor(batch, objective) for batch in evaluation_batches))
    log(
        f'Held-out loss floor: policy {floor.policy:.4f}, wdl {floor.wdl:.4f}, total {floor.total:.4f},'
        f' auxiliary{auxiliary_loss_report(auxiliary_heads, floor) or " (none)"}.'
    )
    log('Headline metric is the held-out policy gap above the policy floor, excluding value and auxiliary losses.')

    generator = np.random.default_rng(arguments.random_seed)
    model.train()
    recent_training_losses: list[LossValues] = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    window_started_at = time.perf_counter()
    window_steps = 0
    for step in range(1, arguments.steps + 1):
        step_learning_rate = learning_rate_at(
            step,
            arguments.steps,
            arguments.learning_rate,
            arguments.warmup_steps,
            arguments.learning_rate_schedule,
            arguments.anneal_fraction,
            arguments.optimizer_kind,
            arguments.floor_fraction,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group['lr'] = step_learning_rate
        batch = sample_training_batch(
            records,
            split.training_row_count,
            arguments.batch_size,
            dataset_manifest.action_size,
            generator,
            device,
            auxiliary_heads,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            loss = objective.calculate_loss(model.training_output(batch.states), batch)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), arguments.max_grad_norm)
        optimizer.step()
        recent_training_losses.append(observed_losses(loss))
        window_steps += 1
        if arguments.checkpoint_every and not step % arguments.checkpoint_every and step != arguments.steps:
            intermediate = intermediate_run_state(arguments.output_run_state, step)
            save_student_checkpoint(model, optimizer, arguments.generation, intermediate)
            log(f'Wrote the step {step} generation {arguments.generation} intermediate student to {intermediate}.')
        if step % arguments.evaluate_every and step != arguments.steps:
            continue
        training_loss = mean_loss_values(tuple(recent_training_losses))
        recent_training_losses.clear()
        window_seconds = time.perf_counter() - window_started_at
        steps_per_second = window_steps / max(window_seconds, 1e-9)
        window_started_at = time.perf_counter()
        window_steps = 0
        peak_mebibytes = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0.0
        held_out_loss = evaluate(model, evaluation_batches, objective, device)
        log(
            f'step {step}/{arguments.steps} lr {step_learning_rate:.2e} '
            f'{steps_per_second:.2f} steps/s {steps_per_second * arguments.batch_size:.0f} samples/s '
            f'peak {peak_mebibytes:.0f} MiB | '
            f'train policy {training_loss.policy:.4f} wdl {training_loss.wdl:.4f} total {training_loss.total:.4f}'
            f'{auxiliary_loss_report(auxiliary_heads, training_loss)} | '
            f'held-out policy {held_out_loss.policy:.4f} wdl {held_out_loss.wdl:.4f} '
            f'total {held_out_loss.total:.4f}{auxiliary_loss_report(auxiliary_heads, held_out_loss)} | '
            f'headline policy gap above floor {held_out_loss.policy - floor.policy:.4f} | '
            f'total gap above floor {held_out_loss.total - floor.total:.4f}'
        )

    manifest = save_student_checkpoint(model, optimizer, arguments.generation, arguments.output_run_state)
    log(f'Wrote generation {arguments.generation} student to {arguments.output_run_state}.')
    return manifest


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Distil a teacher-labelled chess dataset into a small student.')
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output-run-state', required=True, type=Path)
    parser.add_argument(
        '--network-kind',
        default=NetworkKind.CONVOLUTIONAL.value,
        choices=tuple(kind.value for kind in NetworkKind),
    )
    parser.add_argument('--layers', required=True, type=int)
    parser.add_argument('--hidden-size', required=True, type=int)
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--feedforward', default=0, type=int, help='Zero means twice the hidden size.')
    parser.add_argument(
        '--policy-head-kind',
        default=PolicyHeadKind.DENSE.value,
        choices=tuple(kind.value for kind in PolicyHeadKind),
    )
    parser.add_argument('--policy-key-size', default=128, type=int)
    parser.add_argument(
        '--attention-bias-kind',
        default=AttentionBiasKind.DISABLED.value,
        choices=tuple(kind.value for kind in AttentionBiasKind),
    )
    parser.add_argument('--smolgen-compressed-size', default=8, type=int)
    parser.add_argument('--smolgen-hidden-size', default=32, type=int)
    parser.add_argument('--smolgen-generated-size', default=32, type=int)
    parser.add_argument(
        '--optimizer',
        default=OptimizerKind.ADAMW.value,
        choices=tuple(kind.value for kind in OptimizerKind),
    )
    parser.add_argument('--floor-fraction', default=0.1, type=float)
    parser.add_argument(
        '--policy-bottleneck-rank',
        default=16,
        type=int,
        help='Zero removes the bottleneck, which is the dense head production runs.',
    )
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--steps', required=True, type=int)
    parser.add_argument('--learning-rate', default=0.002, type=float)
    parser.add_argument(
        '--learning-rate-schedule',
        default=LearningRateSchedule.COSINE.value,
        choices=tuple(schedule.value for schedule in LearningRateSchedule),
    )
    parser.add_argument('--anneal-fraction', default=0.2, type=float)
    parser.add_argument('--warmup-steps', default=200, type=int)
    parser.add_argument('--max-grad-norm', default=0.5, type=float)
    parser.add_argument('--holdout-fraction', default=0.02, type=float)
    parser.add_argument('--training-fraction', default=1.0, type=float)
    parser.add_argument('--evaluate-every', default=200, type=int)
    parser.add_argument('--checkpoint-every', default=0, type=int)
    parser.add_argument('--distil-auxiliary-heads', nargs='+', default=(), choices=DISTILLABLE_AUXILIARY_HEADS)
    parser.add_argument('--device-id', default=0, type=int)
    parser.add_argument('--random-seed', default=20260826, type=int)
    parser.add_argument('--generation', default=0, type=int)
    namespace = parser.parse_args()
    arguments = Arguments(
        dataset=namespace.dataset,
        output_run_state=namespace.output_run_state,
        network_kind=NetworkKind(namespace.network_kind),
        layers=namespace.layers,
        hidden_size=namespace.hidden_size,
        heads=namespace.heads,
        feedforward=namespace.feedforward or 2 * namespace.hidden_size,
        policy_head_kind=PolicyHeadKind(namespace.policy_head_kind),
        policy_key_size=namespace.policy_key_size,
        attention_bias_kind=AttentionBiasKind(namespace.attention_bias_kind),
        smolgen_compressed_size=namespace.smolgen_compressed_size,
        smolgen_hidden_size=namespace.smolgen_hidden_size,
        smolgen_generated_size=namespace.smolgen_generated_size,
        optimizer_kind=OptimizerKind(namespace.optimizer),
        floor_fraction=namespace.floor_fraction,
        policy_bottleneck_rank=namespace.policy_bottleneck_rank,
        batch_size=namespace.batch_size,
        steps=namespace.steps,
        learning_rate=namespace.learning_rate,
        learning_rate_schedule=LearningRateSchedule(namespace.learning_rate_schedule),
        anneal_fraction=namespace.anneal_fraction,
        warmup_steps=namespace.warmup_steps,
        max_grad_norm=namespace.max_grad_norm,
        holdout_fraction=namespace.holdout_fraction,
        training_fraction=namespace.training_fraction,
        evaluate_every=namespace.evaluate_every,
        checkpoint_every=namespace.checkpoint_every,
        distil_auxiliary_heads=tuple(namespace.distil_auxiliary_heads),
        device_id=namespace.device_id,
        random_seed=namespace.random_seed,
        generation=namespace.generation,
    )
    if not arguments.dataset.is_file():
        raise ValueError(f'Dataset does not exist: {arguments.dataset}')
    if min(arguments.layers, arguments.hidden_size) <= 0:
        raise ValueError('Layers and hidden size must be positive.')
    if arguments.policy_bottleneck_rank < 0:
        raise ValueError('Policy bottleneck rank must be nonnegative; zero removes the bottleneck.')
    if min(arguments.heads, arguments.feedforward, arguments.policy_key_size) <= 0:
        raise ValueError('Head count, feedforward size and policy key size must be positive.')
    if arguments.network_kind is NetworkKind.ATTENTION and arguments.hidden_size % arguments.heads:
        raise ValueError(f'An embedding size of {arguments.hidden_size} is not divisible by {arguments.heads} heads.')
    # The from-to head reads the trunk output as 64 square tokens, which a convolutional trunk also
    # produces, so it is allowed on both: separating the head's contribution from the trunk's needs it.
    if arguments.network_kind is NetworkKind.CONVOLUTIONAL and arguments.attention_bias_kind is not (
        AttentionBiasKind.DISABLED
    ):
        raise ValueError('A convolutional trunk has no attention logits to bias.')
    if min(arguments.smolgen_compressed_size, arguments.smolgen_hidden_size, arguments.smolgen_generated_size) <= 0:
        raise ValueError('Every generated-bias dimension must be positive.')
    if min(arguments.batch_size, arguments.steps, arguments.evaluate_every) <= 0:
        raise ValueError('Batch size, steps and evaluation interval must be positive.')
    if arguments.checkpoint_every < 0:
        raise ValueError('Checkpoint interval must be nonnegative; zero writes the final checkpoint only.')
    if arguments.learning_rate <= 0.0 or arguments.max_grad_norm <= 0.0:
        raise ValueError('Learning rate and gradient-norm bound must be positive.')
    if not 0.0 < arguments.anneal_fraction <= 1.0:
        raise ValueError('Anneal fraction must lie above zero and at most one.')
    if not 0.0 <= arguments.floor_fraction < 1.0:
        raise ValueError('Floor fraction must be nonnegative and below one.')
    if not 0.0 < arguments.holdout_fraction < 1.0:
        raise ValueError('Holdout fraction must lie strictly between zero and one.')
    if not 0.0 < arguments.training_fraction <= 1.0:
        raise ValueError('Training fraction must lie above zero and at most one.')
    if not 0 < arguments.warmup_steps < arguments.steps:
        raise ValueError('Warmup must be positive and shorter than the run.')
    anneal_start = arguments.steps - max(1, round(arguments.steps * arguments.anneal_fraction))
    if arguments.learning_rate_schedule is LearningRateSchedule.PLATEAU and arguments.warmup_steps >= anneal_start:
        raise ValueError(
            f'A warmup of {arguments.warmup_steps} steps leaves no plateau before the anneal starts at step '
            f'{anneal_start}.'
        )
    if len(set(arguments.distil_auxiliary_heads)) != len(arguments.distil_auxiliary_heads):
        raise ValueError('Each distilled auxiliary head may be named at most once.')
    captured_heads = read_manifest(arguments.dataset).captured_auxiliary_heads
    absent_heads = tuple(head for head in arguments.distil_auxiliary_heads if head not in captured_heads)
    if absent_heads:
        raise ValueError(
            f'Dataset {arguments.dataset} captured auxiliary heads {captured_heads} and cannot supply {absent_heads}.'
        )
    if arguments.device_id < 0 or arguments.random_seed < 0 or arguments.generation < 0:
        raise ValueError('Device ID, random seed and generation must be nonnegative.')
    return arguments


def main() -> None:
    train_student(parse_arguments())


if __name__ == '__main__':
    main()
