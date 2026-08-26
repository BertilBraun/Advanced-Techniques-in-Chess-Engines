from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from src.distillation.dataset import build_training_batch, open_dataset
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS, CHESS_STATE_CONTRACT
from src.training.batch import TrainingBatch
from src.training.checkpoint.contracts import CheckpointManifest
from src.training.checkpoint.paths import checkpoint_manifest_path, model_save_path, optimizer_save_path
from src.training.checkpoint.persistence import create_model, create_optimizer
from src.training.network import (
    DensePolicyHeadConfiguration,
    GlobalPoolingResidualContext,
    InferenceNetwork,
    Network,
    NetworkParams,
    ResidualContextPlacement,
)
from src.training.objective import ObjectiveLoss, ResolvedTrainingObjective
from src.util.atomic_file import write_text_atomically
from src.util.hashing import file_sha256
from src.util.log import log
from tools.benchmark_training_overfit import LossValues, achievable_loss_floor

HELD_OUT_EVALUATION_BATCHES = 8


@dataclass(frozen=True)
class Arguments:
    dataset: Path
    output_run_state: Path
    layers: int
    hidden_size: int
    policy_bottleneck_rank: int
    batch_size: int
    steps: int
    learning_rate: float
    warmup_steps: int
    max_grad_norm: float
    holdout_fraction: float
    evaluate_every: int
    device_id: int
    random_seed: int
    generation: int


@dataclass(frozen=True)
class ParameterCounts:
    backbone: int
    heads: int

    @property
    def total(self) -> int:
        return self.backbone + self.heads


def student_architecture(layers: int, hidden_size: int, policy_bottleneck_rank: int) -> NetworkParams:
    return NetworkParams(
        num_layers=layers,
        hidden_size=hidden_size,
        residual_context=GlobalPoolingResidualContext(placement=ResidualContextPlacement.EVERY_SECOND_BLOCK),
        policy_head=DensePolicyHeadConfiguration(channels=4, bottleneck_rank=policy_bottleneck_rank),
        num_value_channels=2,
        value_fc_size=48,
    )


def distillation_objective() -> ResolvedTrainingObjective:
    return ResolvedTrainingObjective(
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        root_value_blend=0.0,
        auxiliary_losses=(),
    )


def parameter_counts(model: Network) -> ParameterCounts:
    def count(*modules: torch.nn.Module) -> int:
        return sum(parameter.numel() for module in modules for parameter in module.parameters())

    return ParameterCounts(
        backbone=count(model.start_block, model.backbone, model.finish_block),
        heads=count(model.policy_head, model.value_head, model.auxiliary_head_modules),
    )


def learning_rate_at(step: int, total_steps: int, peak_learning_rate: float, warmup_steps: int) -> float:
    if step <= warmup_steps:
        return peak_learning_rate * step / warmup_steps
    decay_progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return peak_learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(decay_progress, 1.0)))


def observed_losses(loss: ObjectiveLoss) -> LossValues:
    return LossValues(
        policy=float(loss.policy.detach()),
        wdl=float(loss.wdl.detach()),
        auxiliary=(),
        total=float(loss.total.detach()),
    )


def mean_loss_values(values: tuple[LossValues, ...]) -> LossValues:
    return LossValues(
        policy=sum(value.policy for value in values) / len(values),
        wdl=sum(value.wdl for value in values) / len(values),
        auxiliary=(),
        total=sum(value.total for value in values) / len(values),
    )


def sample_training_batch(
    records: npt.NDArray,
    training_row_count: int,
    batch_size: int,
    action_size: int,
    generator: np.random.Generator,
    device: torch.device,
) -> TrainingBatch:
    indices = np.sort(generator.integers(0, training_row_count, size=batch_size))
    return build_training_batch(records[indices], CHESS_STATE_CONTRACT, action_size, device)


def held_out_batches(
    records: npt.NDArray,
    training_row_count: int,
    batch_size: int,
    action_size: int,
    device: torch.device,
) -> tuple[TrainingBatch, ...]:
    available_rows = len(records) - training_row_count
    rows_per_batch = min(batch_size, available_rows)
    batch_count = min(HELD_OUT_EVALUATION_BATCHES, available_rows // rows_per_batch)
    return tuple(
        build_training_batch(
            records[training_row_count + index * rows_per_batch : training_row_count + (index + 1) * rows_per_batch],
            CHESS_STATE_CONTRACT,
            action_size,
            device,
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

    held_out_row_count = max(1, round(len(records) * arguments.holdout_fraction))
    training_row_count = len(records) - held_out_row_count
    if training_row_count < arguments.batch_size:
        raise ValueError(f'{training_row_count} training rows are fewer than one batch of {arguments.batch_size}.')

    device = select_device(arguments.device_id)
    torch.manual_seed(arguments.random_seed)
    torch.cuda.manual_seed_all(arguments.random_seed)

    architecture = student_architecture(arguments.layers, arguments.hidden_size, arguments.policy_bottleneck_rank)
    model = create_model(architecture, device, CHESS_NETWORK_DIMENSIONS)
    optimizer = create_optimizer(model, 'adamw')
    objective = distillation_objective()
    counts = parameter_counts(model)
    log(f'Student parameters: {counts.total} total = {counts.backbone} backbone + {counts.heads} heads.')
    log(f'Training on {training_row_count} rows, holding out the last {held_out_row_count} rows.')

    evaluation_batches = held_out_batches(
        records,
        training_row_count,
        arguments.batch_size,
        dataset_manifest.action_size,
        device,
    )
    floor = mean_loss_values(tuple(achievable_loss_floor(batch, objective) for batch in evaluation_batches))
    log(f'Held-out loss floor: policy {floor.policy:.4f}, wdl {floor.wdl:.4f}, total {floor.total:.4f}.')

    generator = np.random.default_rng(arguments.random_seed)
    model.train()
    recent_training_losses: list[LossValues] = []
    for step in range(1, arguments.steps + 1):
        step_learning_rate = learning_rate_at(step, arguments.steps, arguments.learning_rate, arguments.warmup_steps)
        for parameter_group in optimizer.param_groups:
            parameter_group['lr'] = step_learning_rate
        batch = sample_training_batch(
            records,
            training_row_count,
            arguments.batch_size,
            dataset_manifest.action_size,
            generator,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            loss = objective.calculate_loss(model.training_output(batch.states), batch)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), arguments.max_grad_norm)
        optimizer.step()
        recent_training_losses.append(observed_losses(loss))
        if step % arguments.evaluate_every and step != arguments.steps:
            continue
        training_loss = mean_loss_values(tuple(recent_training_losses))
        recent_training_losses.clear()
        held_out_loss = evaluate(model, evaluation_batches, objective, device)
        log(
            f'step {step}/{arguments.steps} lr {step_learning_rate:.2e} '
            f'train policy {training_loss.policy:.4f} wdl {training_loss.wdl:.4f} total {training_loss.total:.4f} | '
            f'held-out policy {held_out_loss.policy:.4f} wdl {held_out_loss.wdl:.4f} '
            f'total {held_out_loss.total:.4f} | '
            f'distillation gap {held_out_loss.policy - floor.policy:.4f}'
        )

    manifest = save_student_checkpoint(model, optimizer, arguments.generation, arguments.output_run_state)
    log(f'Wrote generation {arguments.generation} student to {arguments.output_run_state}.')
    return manifest


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Distil a teacher-labelled chess dataset into a small student.')
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output-run-state', required=True, type=Path)
    parser.add_argument('--layers', required=True, type=int)
    parser.add_argument('--hidden-size', required=True, type=int)
    parser.add_argument('--policy-bottleneck-rank', default=16, type=int)
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--steps', required=True, type=int)
    parser.add_argument('--learning-rate', default=0.002, type=float)
    parser.add_argument('--warmup-steps', default=200, type=int)
    parser.add_argument('--max-grad-norm', default=0.5, type=float)
    parser.add_argument('--holdout-fraction', default=0.02, type=float)
    parser.add_argument('--evaluate-every', default=200, type=int)
    parser.add_argument('--device-id', default=0, type=int)
    parser.add_argument('--random-seed', default=20260826, type=int)
    parser.add_argument('--generation', default=0, type=int)
    namespace = parser.parse_args()
    arguments = Arguments(
        dataset=namespace.dataset,
        output_run_state=namespace.output_run_state,
        layers=namespace.layers,
        hidden_size=namespace.hidden_size,
        policy_bottleneck_rank=namespace.policy_bottleneck_rank,
        batch_size=namespace.batch_size,
        steps=namespace.steps,
        learning_rate=namespace.learning_rate,
        warmup_steps=namespace.warmup_steps,
        max_grad_norm=namespace.max_grad_norm,
        holdout_fraction=namespace.holdout_fraction,
        evaluate_every=namespace.evaluate_every,
        device_id=namespace.device_id,
        random_seed=namespace.random_seed,
        generation=namespace.generation,
    )
    if not arguments.dataset.is_file():
        raise ValueError(f'Dataset does not exist: {arguments.dataset}')
    if min(arguments.layers, arguments.hidden_size, arguments.policy_bottleneck_rank) <= 0:
        raise ValueError('Layers, hidden size and policy bottleneck rank must be positive.')
    if min(arguments.batch_size, arguments.steps, arguments.evaluate_every) <= 0:
        raise ValueError('Batch size, steps and evaluation interval must be positive.')
    if arguments.learning_rate <= 0.0 or arguments.max_grad_norm <= 0.0:
        raise ValueError('Learning rate and gradient-norm bound must be positive.')
    if not 0.0 < arguments.holdout_fraction < 1.0:
        raise ValueError('Holdout fraction must lie strictly between zero and one.')
    if not 0 <= arguments.warmup_steps < arguments.steps:
        raise ValueError('Warmup must be nonnegative and shorter than the run.')
    if arguments.device_id < 0 or arguments.random_seed < 0 or arguments.generation < 0:
        raise ValueError('Device ID, random seed and generation must be nonnegative.')
    return arguments


def main() -> None:
    train_student(parse_arguments())


if __name__ == '__main__':
    main()
