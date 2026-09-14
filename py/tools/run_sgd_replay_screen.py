from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as distributed
from modelopt.torch.quantization.nn import TensorQuantizer
from pydantic import Field
from src.distillation.dataset import build_replay_training_batch
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS, CHESS_STATE_CONTRACT
from src.training.batch import TrainingBatch
from src.training.checkpoint.persistence import create_optimizer
from src.training.configuration import SgdOptimizerConfiguration
from src.training.network import Network
from src.training.objective import ResolvedTrainingObjective, mask_policy_logits
from src.training.quantization.runtime import (
    configure_qat,
    fold_scaled_post_activation_batch_norm,
    recalibrate_qat,
)
from src.training.trainer.rank import DistributedTrainingModel
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from tools.distill_train_student import (
    OpenedProductionReplay,
    ProductionReplayInput,
    close_training_dataset,
    dataset_split,
    distillation_objective,
    open_training_dataset,
)
from tools.run_int8_architecture_screen import ScreenCell, architecture
from torch.nn.parallel import DistributedDataParallel


class ScreenArm(str, Enum):
    V35_CONTROL = 'v35_control'
    LR_004_WARM_2000 = 'lr_004_warm_2000'
    LR_006_WARM_3000 = 'lr_006_warm_3000'
    HISTORICAL_FOLD_1000 = 'historical_fold_1000'
    CONTINUOUS_FOLD_1000 = 'continuous_fold_1000'
    HISTORICAL_FOLD_3000 = 'historical_fold_3000'
    CONTINUOUS_FOLD_3000 = 'continuous_fold_3000'


@dataclass(frozen=True)
class ArmSchedule:
    pre_fold_peak_learning_rate: float
    pre_fold_warmup_start_learning_rate: float
    pre_fold_warmup_steps: int
    fold_after_optimizer_steps: int
    deployment_learning_rate: float
    deployment_warmup_start_learning_rate: float
    deployment_warmup_steps: int


ARM_SCHEDULES = {
    ScreenArm.V35_CONTROL: ArmSchedule(0.1, 0.0, 1_000, 1_000, 0.02, 0.001, 0),
    ScreenArm.LR_004_WARM_2000: ArmSchedule(0.1, 0.0, 1_000, 1_000, 0.04, 0.001, 2_000),
    ScreenArm.LR_006_WARM_3000: ArmSchedule(0.1, 0.0, 1_000, 1_000, 0.06, 0.001, 3_000),
    ScreenArm.HISTORICAL_FOLD_1000: ArmSchedule(0.1, 0.0, 1_000, 1_000, 0.02, 0.0, 0),
    ScreenArm.CONTINUOUS_FOLD_1000: ArmSchedule(0.02, 0.0001, 1_000, 1_000, 0.02, 0.0, 0),
    ScreenArm.HISTORICAL_FOLD_3000: ArmSchedule(0.1, 0.0, 1_000, 3_000, 0.02, 0.0, 0),
    ScreenArm.CONTINUOUS_FOLD_3000: ArmSchedule(0.02, 0.0001, 1_000, 3_000, 0.02, 0.0, 0),
}


class LossMetrics(FrozenModel):
    policy: float
    wdl: float
    total: float


class QuantizerRangeMetrics(FrozenModel):
    tensor_count: int = Field(gt=0)
    minimum: float = Field(ge=0.0)
    median: float = Field(ge=0.0)
    p95: float = Field(ge=0.0)
    maximum: float = Field(ge=0.0)


class Observation(FrozenModel):
    optimizer_step: int = Field(ge=0)
    phase: Literal['pre_fold', 'deployment']
    actual_learning_rate: float = Field(ge=0.0)
    elapsed_seconds: float = Field(ge=0.0)
    interval_samples_per_second: float | None = Field(default=None, gt=0.0)
    training: LossMetrics | None
    held_out: LossMetrics
    held_out_target_top_action_agreement: float = Field(ge=0.0, le=1.0)
    mean_gradient_norm: float | None = Field(default=None, ge=0.0)
    maximum_gradient_norm: float | None = Field(default=None, ge=0.0)
    clipped_step_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    activation_ranges: QuantizerRangeMetrics
    weight_ranges: QuantizerRangeMetrics


class ScreenReport(FrozenModel):
    schema_version: Literal[2] = 2
    arm: ScreenArm
    world_size: Literal[2] = 2
    gpu_ids: tuple[int, int]
    random_seed: int = Field(ge=0)
    replay_store: Path
    replay_store_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    replay_experiment: Path
    replay_experiment_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    replay_rows: int = Field(gt=0)
    training_rows: int = Field(gt=0)
    held_out_start_row: int = Field(gt=0)
    held_out_positions: int = Field(gt=0)
    sampled_global_index_sequence_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    initial_state_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    final_state_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    global_batch_size: Literal[2048] = 2048
    local_batch_size: Literal[1024] = 1024
    optimizer: SgdOptimizerConfiguration
    maximum_gradient_norm: Literal[1.0] = 1.0
    pre_fold_peak_learning_rate: float = Field(gt=0.0)
    pre_fold_warmup_start_learning_rate: float = Field(ge=0.0)
    pre_fold_warmup_steps: int = Field(gt=0)
    fold_after_optimizer_steps: int = Field(gt=0)
    deployment_learning_rate: float = Field(gt=0.0)
    deployment_warmup_start_learning_rate: float = Field(ge=0.0)
    deployment_warmup_steps: int = Field(ge=0)
    completed_optimizer_steps: int = Field(ge=0)
    wall_seconds: float = Field(ge=0.0)
    diverged: bool
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class Arguments:
    arm: ScreenArm
    replay_store: Path
    replay_experiment: Path
    replay_sha256: str
    output: Path
    gpu_ids: tuple[int, int]
    random_seed: int
    time_budget_seconds: float
    maximum_optimizer_steps: int
    held_out_positions: int
    holdout_fraction: float


def _batch(dataset: OpenedProductionReplay, indices: np.ndarray, device: torch.device) -> TrainingBatch:
    return build_replay_training_batch(
        dataset.store.gather_logical(indices), CHESS_STATE_CONTRACT, dataset.action_size, device
    )


def _calibration_loop(
    dataset: OpenedProductionReplay, indices: np.ndarray, device: torch.device
) -> Callable[[torch.nn.Module], None]:
    def calibrate(model: torch.nn.Module) -> None:
        was_training = model.training
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(indices), 64):
                model(_batch(dataset, indices[start : start + 64], device).states)
        model.train(was_training)

    return calibrate


def _loss_metrics(policy: float, wdl: float, total: float) -> LossMetrics:
    return LossMetrics(policy=policy, wdl=wdl, total=total)


def _range_metrics(values: tuple[torch.Tensor, ...]) -> QuantizerRangeMetrics:
    flattened = torch.cat(tuple(value.detach().float().flatten().cpu() for value in values))
    return QuantizerRangeMetrics(
        tensor_count=len(values),
        minimum=float(flattened.min()),
        median=float(torch.quantile(flattened, 0.5)),
        p95=float(torch.quantile(flattened, 0.95)),
        maximum=float(flattened.max()),
    )


def _quantizer_ranges(model: Network) -> tuple[QuantizerRangeMetrics, QuantizerRangeMetrics]:
    activation_ranges: list[torch.Tensor] = []
    weight_ranges: list[torch.Tensor] = []
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer) or not module.is_enabled:
            continue
        if name.endswith('weight_quantizer'):
            weight_ranges.append(module.amax)
        else:
            activation_ranges.append(module.amax)
    return _range_metrics(tuple(activation_ranges)), _range_metrics(tuple(weight_ranges))


def _evaluate(
    model: Network,
    dataset: OpenedProductionReplay,
    held_out_start: int,
    held_out_positions: int,
    objective: ResolvedTrainingObjective,
    device: torch.device,
) -> tuple[LossMetrics, float]:
    model.eval()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for start in range(held_out_start, held_out_start + held_out_positions, 256):
            indices = np.arange(start, min(start + 256, held_out_start + held_out_positions), dtype=np.int64)
            batch = _batch(dataset, indices, device)
            output = model.training_output(batch.states)
            loss = objective.calculate_loss(output, batch)
            predicted = mask_policy_logits(output.policy_logits, batch.policy_legal_action_ids).argmax(dim=1)
            rows = len(batch)
            totals += torch.tensor(
                (
                    float(loss.policy) * rows,
                    float(loss.wdl) * rows,
                    float(loss.total) * rows,
                    float((predicted == batch.policy_targets.argmax(dim=1)).sum()),
                ),
                dtype=torch.float64,
                device=device,
            )
    model.train()
    values = totals.cpu().numpy() / held_out_positions
    return _loss_metrics(float(values[0]), float(values[1]), float(values[2])), float(values[3])


def _learning_rate(step: int, schedule: ArmSchedule) -> float:
    if step <= schedule.pre_fold_warmup_steps:
        progress = step / schedule.pre_fold_warmup_steps
        return schedule.pre_fold_warmup_start_learning_rate + (
            schedule.pre_fold_peak_learning_rate - schedule.pre_fold_warmup_start_learning_rate
        ) * progress
    if step <= schedule.fold_after_optimizer_steps:
        return schedule.pre_fold_peak_learning_rate
    deployment_step = step - schedule.fold_after_optimizer_steps
    if schedule.deployment_warmup_steps == 0:
        return schedule.deployment_learning_rate
    progress = min(deployment_step / schedule.deployment_warmup_steps, 1.0)
    return schedule.deployment_warmup_start_learning_rate + (
        schedule.deployment_learning_rate - schedule.deployment_warmup_start_learning_rate
    ) * progress


def _phase(step: int, schedule: ArmSchedule) -> Literal['pre_fold', 'deployment']:
    return 'pre_fold' if step < schedule.fold_after_optimizer_steps else 'deployment'


def _save_state(model: Network, path: Path) -> str:
    torch.save(model.state_dict(), path)
    return file_sha256(path)


def _write_csv(path: Path, observations: tuple[Observation, ...]) -> None:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(Observation.model_fields))
        writer.writeheader()
        for observation in observations:
            row = observation.model_dump(mode='json')
            row['training'] = json.dumps(row['training'], sort_keys=True)
            row['held_out'] = json.dumps(row['held_out'], sort_keys=True)
            writer.writerow(row)


def run(arguments: Arguments) -> None:
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if world_size != 2:
        raise ValueError('The SGD replay screen requires exactly two DDP ranks.')
    device_id = arguments.gpu_ids[local_rank]
    device = torch.device('cuda', device_id)
    torch.cuda.set_device(device)
    distributed.init_process_group(backend='nccl')
    dataset_input = ProductionReplayInput(
        kind='production_replay',
        path=arguments.replay_store,
        experiment=arguments.replay_experiment,
        orchestrator_recorded_sha256=arguments.replay_sha256,
    )
    opened = open_training_dataset(dataset_input)
    if not isinstance(opened, OpenedProductionReplay):
        raise AssertionError('The SGD replay screen requires a production replay store.')
    try:
        split = dataset_split(opened.row_count, arguments.holdout_fraction, 1.0)
        if arguments.held_out_positions > split.held_out_row_count:
            raise ValueError('Requested held-out positions exceed the fixed replay tail.')
        torch.manual_seed(arguments.random_seed)
        torch.cuda.manual_seed_all(arguments.random_seed)
        model = Network(architecture(ScreenCell.POST_SCALED_QAT), device, CHESS_NETWORK_DIMENSIONS)
        calibration_generator = np.random.default_rng(arguments.random_seed + 10_000_000)
        calibration_indices = np.sort(calibration_generator.choice(split.training_row_count, size=256, replace=False))
        model = configure_qat(model, _calibration_loop(opened, calibration_indices, device))
        optimizer_configuration = SgdOptimizerConfiguration(momentum=0.9, weight_decay=0.0001, nesterov=True)
        optimizer = create_optimizer(model, optimizer_configuration)
        distributed_model = DistributedDataParallel(
            DistributedTrainingModel(model), device_ids=[device_id], broadcast_buffers=False
        )
        objective = distillation_objective()
        schedule = ARM_SCHEDULES[arguments.arm]
        arguments.output.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            initial_state_sha256 = _save_state(model, arguments.output / 'initial-state.pt')
        else:
            initial_state_sha256 = ''
        distributed.barrier()

        sample_generator = np.random.default_rng(arguments.random_seed)
        sampled_indices_hash = hashlib.sha256()
        observations: list[Observation] = []
        recent_loss_totals = np.zeros(3)
        recent_gradient_total = 0.0
        recent_gradient_maximum = 0.0
        recent_clipped_steps = 0
        recent_steps = 0
        interval_started = time.perf_counter()
        started = interval_started
        completed_steps = 0
        diverged = False
        report_steps = set(range(500, arguments.maximum_optimizer_steps + 1, 500))
        report_steps.update(
            step
            for step in (
                schedule.fold_after_optimizer_steps - 1,
                schedule.fold_after_optimizer_steps,
                schedule.fold_after_optimizer_steps + 1,
            )
            if step > 0
        )

        if rank == 0:
            held_out, agreement = _evaluate(
                model, opened, split.held_out_start_row, arguments.held_out_positions, objective, device
            )
            activation_ranges, weight_ranges = _quantizer_ranges(model)
            observations.append(
                Observation(
                    optimizer_step=0,
                    phase='pre_fold',
                    actual_learning_rate=0.0,
                    elapsed_seconds=time.perf_counter() - started,
                    held_out=held_out,
                    held_out_target_top_action_agreement=agreement,
                    training=None,
                    activation_ranges=activation_ranges,
                    weight_ranges=weight_ranges,
                )
            )
        distributed.barrier()

        for step in range(1, arguments.maximum_optimizer_steps + 1):
            global_indices = np.sort(sample_generator.integers(0, split.training_row_count, size=2_048, dtype=np.int64))
            if rank == 0:
                sampled_indices_hash.update(global_indices.tobytes())
            local_indices = global_indices[local_rank * 1_024 : (local_rank + 1) * 1_024]
            batch = _batch(opened, local_indices, device)
            actual_learning_rate = _learning_rate(step, schedule)
            for parameter_group in optimizer.param_groups:
                parameter_group['lr'] = actual_learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = distributed_model(batch.states)
                loss = objective.calculate_loss(output, batch)
            loss.total.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(distributed_model.parameters(), 1.0))
            optimizer.step()
            local_metrics = torch.stack((loss.policy.detach(), loss.wdl.detach(), loss.total.detach())).to(
                dtype=torch.float64
            )
            distributed.all_reduce(local_metrics, op=distributed.ReduceOp.SUM)
            metrics = local_metrics.cpu().numpy() / world_size
            recent_loss_totals += metrics
            recent_gradient_total += gradient_norm
            recent_gradient_maximum = max(recent_gradient_maximum, gradient_norm)
            recent_clipped_steps += int(gradient_norm > 1.0)
            recent_steps += 1
            completed_steps = step
            if not all(math.isfinite(value) for value in (*metrics, gradient_norm)):
                diverged = True

            if step == schedule.fold_after_optimizer_steps:
                distributed.barrier()
                del distributed_model
                fold_scaled_post_activation_batch_norm(model)
                recalibrate_qat(model, _calibration_loop(opened, calibration_indices, device))
                optimizer = create_optimizer(model, optimizer_configuration)
                distributed_model = DistributedDataParallel(
                    DistributedTrainingModel(model), device_ids=[device_id], broadcast_buffers=False
                )
                distributed.barrier()
            elif step % 500 == 0:
                distributed.barrier()
                recalibrate_qat(model, _calibration_loop(opened, calibration_indices, device))
                distributed.barrier()

            should_report = step in report_steps or diverged
            if should_report:
                distributed.barrier()
                if rank == 0:
                    now = time.perf_counter()
                    held_out, agreement = _evaluate(
                        model, opened, split.held_out_start_row, arguments.held_out_positions, objective, device
                    )
                    activation_ranges, weight_ranges = _quantizer_ranges(model)
                    observations.append(
                        Observation(
                            optimizer_step=step,
                            phase=_phase(step, schedule),
                            actual_learning_rate=actual_learning_rate,
                            elapsed_seconds=now - started,
                            interval_samples_per_second=recent_steps * 2_048 / (now - interval_started),
                            training=_loss_metrics(*(recent_loss_totals / recent_steps)),
                            held_out=held_out,
                            held_out_target_top_action_agreement=agreement,
                            mean_gradient_norm=recent_gradient_total / recent_steps,
                            maximum_gradient_norm=recent_gradient_maximum,
                            clipped_step_fraction=recent_clipped_steps / recent_steps,
                            activation_ranges=activation_ranges,
                            weight_ranges=weight_ranges,
                        )
                    )
                    print(observations[-1].model_dump_json(), flush=True)
                distributed.barrier()
                recent_loss_totals.fill(0.0)
                recent_gradient_total = 0.0
                recent_gradient_maximum = 0.0
                recent_clipped_steps = 0
                recent_steps = 0
                interval_started = time.perf_counter()

            stop_tensor = torch.zeros(1, dtype=torch.uint8, device=device)
            if step % 25 == 0 or diverged:
                if rank == 0 and (diverged or time.perf_counter() - started >= arguments.time_budget_seconds):
                    stop_tensor.fill_(1)
                distributed.broadcast(stop_tensor, src=0)
                if bool(stop_tensor.item()):
                    break

        distributed.barrier()
        if rank == 0:
            if not observations or observations[-1].optimizer_step != completed_steps:
                now = time.perf_counter()
                held_out, agreement = _evaluate(
                    model, opened, split.held_out_start_row, arguments.held_out_positions, objective, device
                )
                activation_ranges, weight_ranges = _quantizer_ranges(model)
                observations.append(
                    Observation(
                        optimizer_step=completed_steps,
                        phase=_phase(completed_steps, schedule),
                        actual_learning_rate=_learning_rate(completed_steps, schedule),
                        elapsed_seconds=now - started,
                        interval_samples_per_second=(
                            recent_steps * 2_048 / (now - interval_started) if recent_steps else None
                        ),
                        training=(_loss_metrics(*(recent_loss_totals / recent_steps)) if recent_steps else None),
                        held_out=held_out,
                        held_out_target_top_action_agreement=agreement,
                        mean_gradient_norm=recent_gradient_total / recent_steps if recent_steps else None,
                        maximum_gradient_norm=recent_gradient_maximum if recent_steps else None,
                        clipped_step_fraction=recent_clipped_steps / recent_steps if recent_steps else None,
                        activation_ranges=activation_ranges,
                        weight_ranges=weight_ranges,
                    )
                )
            final_state_sha256 = _save_state(model, arguments.output / 'final-state.pt')
            report = ScreenReport(
                arm=arguments.arm,
                gpu_ids=arguments.gpu_ids,
                random_seed=arguments.random_seed,
                replay_store=arguments.replay_store.resolve(),
                replay_store_sha256=arguments.replay_sha256,
                replay_experiment=arguments.replay_experiment.resolve(),
                replay_experiment_sha256=file_sha256(arguments.replay_experiment),
                replay_rows=opened.row_count,
                training_rows=split.training_row_count,
                held_out_start_row=split.held_out_start_row,
                held_out_positions=arguments.held_out_positions,
                sampled_global_index_sequence_sha256=sampled_indices_hash.hexdigest(),
                initial_state_sha256=initial_state_sha256,
                final_state_sha256=final_state_sha256,
                optimizer=optimizer_configuration,
                pre_fold_peak_learning_rate=schedule.pre_fold_peak_learning_rate,
                pre_fold_warmup_start_learning_rate=schedule.pre_fold_warmup_start_learning_rate,
                pre_fold_warmup_steps=schedule.pre_fold_warmup_steps,
                fold_after_optimizer_steps=schedule.fold_after_optimizer_steps,
                deployment_learning_rate=schedule.deployment_learning_rate,
                deployment_warmup_start_learning_rate=schedule.deployment_warmup_start_learning_rate,
                deployment_warmup_steps=schedule.deployment_warmup_steps,
                completed_optimizer_steps=completed_steps,
                wall_seconds=time.perf_counter() - started,
                diverged=diverged,
                observations=tuple(observations),
            )
            write_text_atomically(arguments.output / 'report.json', report.model_dump_json(indent=2) + '\n')
            _write_csv(arguments.output / 'observations.csv', report.observations)
        distributed.barrier()
    finally:
        close_training_dataset(opened)
        distributed.destroy_process_group()


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Run one matched two-GPU SGD QAT frozen-replay screen arm.')
    parser.add_argument('--arm', required=True, choices=tuple(arm.value for arm in ScreenArm))
    parser.add_argument('--replay-store', required=True, type=Path)
    parser.add_argument('--replay-experiment', required=True, type=Path)
    parser.add_argument('--replay-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-ids', required=True, nargs=2, type=int)
    parser.add_argument('--random-seed', default=20260913, type=int)
    parser.add_argument('--time-budget-seconds', default=1_200.0, type=float)
    parser.add_argument('--maximum-optimizer-steps', default=12_000, type=int)
    parser.add_argument('--held-out-positions', default=4_096, type=int)
    parser.add_argument('--holdout-fraction', default=0.02, type=float)
    namespace = parser.parse_args()
    gpu_ids = tuple(namespace.gpu_ids)
    if len(set(gpu_ids)) != 2 or min(gpu_ids) < 0:
        raise ValueError('The arm requires two distinct nonnegative GPU IDs.')
    if namespace.time_budget_seconds <= 0 or namespace.maximum_optimizer_steps <= 1_000:
        raise ValueError('The screen must run past the fold with a positive time budget.')
    return Arguments(
        arm=ScreenArm(namespace.arm),
        replay_store=namespace.replay_store,
        replay_experiment=namespace.replay_experiment,
        replay_sha256=namespace.replay_sha256,
        output=namespace.output,
        gpu_ids=gpu_ids,
        random_seed=namespace.random_seed,
        time_budget_seconds=namespace.time_budget_seconds,
        maximum_optimizer_steps=namespace.maximum_optimizer_steps,
        held_out_positions=namespace.held_out_positions,
        holdout_fraction=namespace.holdout_fraction,
    )


if __name__ == '__main__':
    run(parse_arguments())
