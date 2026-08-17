from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Literal

import numpy as np
import torch
import torch.distributed as distributed
from pydantic import Field
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from src.training.architecture_benchmark import (
    ArchitectureBenchmarkPlan,
    ComparisonProtocol,
    load_architecture_benchmark_plan,
)
from src.training.architecture_catalog import ArchitectureCatalogEntry, load_architecture_catalog
from src.training.network import Network
from src.util.frozen_model import FrozenModel


@dataclass(frozen=True)
class FrozenReplayDataset:
    states: Tensor
    policy_targets: Tensor
    wdl_targets: Tensor
    next_policy_targets: Tensor
    remaining_length_targets: Tensor

    @property
    def sample_count(self) -> int:
        return self.states.shape[0]


class InferenceMeasurement(FrozenModel):
    batch_size: int
    mean_seconds: float
    minimum_rank_positions_per_second: float
    aggregate_positions_per_second: float
    maximum_rank_peak_allocated_bytes: int
    maximum_rank_peak_reserved_bytes: int


class TrainingMeasurement(FrozenModel):
    protocol: ComparisonProtocol
    optimizer_steps: int
    global_samples: int
    elapsed_seconds: float
    samples_per_second: float
    maximum_rank_peak_allocated_bytes: int
    maximum_rank_peak_reserved_bytes: int


class ArchitectureBenchmarkResult(FrozenModel):
    schema_version: Literal[1] = 1
    model_id: str
    parameter_count: int = Field(gt=0)
    world_size: int = Field(gt=0)
    plan: ArchitectureBenchmarkPlan
    frozen_replay_sha256: str
    torch_version: str
    cuda_version: str
    device_name: str
    training: TrainingMeasurement
    inference: tuple[InferenceMeasurement, ...]


class TrainingNetwork(nn.Module):
    def __init__(self, network: Network) -> None:
        super().__init__()
        self.network = network

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
        output = self.network.training_output(states)
        return output.policy_logits, output.wdl_logits, output.auxiliary_logits


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Explicitly gated chess architecture GPU benchmark harness.')
    parser.add_argument(
        '--plan',
        type=Path,
        default=Path('configs/benchmarks/chess-architecture-v1.yaml'),
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('describe')

    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('model_id')
    run_parser.add_argument('--protocol', type=ComparisonProtocol, required=True)
    run_parser.add_argument('--frozen-replay', type=Path, required=True)
    run_parser.add_argument('--output', type=Path, required=True)
    run_parser.add_argument('--acknowledge-gpu-load', action='store_true')
    return parser.parse_args()


def _load_frozen_replay(path: Path) -> FrozenReplayDataset:
    with np.load(path, allow_pickle=False) as archive:
        dataset = FrozenReplayDataset(
            states=torch.from_numpy(archive['states'].copy()).float(),
            policy_targets=torch.from_numpy(archive['policy_targets'].copy()).float(),
            wdl_targets=torch.from_numpy(archive['wdl_targets'].copy()).float(),
            next_policy_targets=torch.from_numpy(archive['next_policy_targets'].copy()).float(),
            remaining_length_targets=torch.from_numpy(archive['remaining_length_targets'].copy()).float(),
        )
    lengths = {
        dataset.states.shape[0],
        dataset.policy_targets.shape[0],
        dataset.wdl_targets.shape[0],
        dataset.next_policy_targets.shape[0],
        dataset.remaining_length_targets.shape[0],
    }
    if len(lengths) != 1:
        raise ValueError('Frozen replay arrays must contain the same number of samples.')
    if dataset.states.ndim != 4:
        raise ValueError('Frozen replay states must have batch, channel, row, and column dimensions.')
    if dataset.remaining_length_targets.ndim != 2 or dataset.remaining_length_targets.shape[1] != 1:
        raise ValueError('Frozen remaining-length targets must have shape (samples, 1).')
    return dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_entry(plan: ArchitectureBenchmarkPlan, model_id: str) -> ArchitectureCatalogEntry:
    catalog = load_architecture_catalog(plan.catalog_path)
    matches = tuple(entry for entry in catalog.models if entry.model_id == model_id)
    if len(matches) != 1:
        raise ValueError(f'Architecture catalog must contain exactly one model named {model_id}.')
    return matches[0]


def _distributed_context(plan: ArchitectureBenchmarkPlan) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise ValueError('Architecture benchmark requires CUDA.')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if world_size != len(plan.topology.trainer_device_ids):
        raise ValueError('Torchrun world size does not match the benchmark production topology.')
    device_id = plan.topology.trainer_device_ids[local_rank]
    torch.cuda.set_device(device_id)
    distributed.init_process_group('nccl')
    return rank, world_size, torch.device('cuda', device_id)


def _batch_indices(
    dataset_size: int,
    global_step: int,
    rank: int,
    plan: ArchitectureBenchmarkPlan,
) -> Tensor:
    local_batch_size = plan.topology.local_training_batch_size
    start = global_step * plan.topology.global_training_batch_size + rank * local_batch_size
    return torch.arange(start, start + local_batch_size).remainder(dataset_size)


def _training_loss(
    outputs: tuple[Tensor, Tensor, tuple[Tensor, ...]],
    dataset: FrozenReplayDataset,
    indices: Tensor,
    device: torch.device,
) -> Tensor:
    policy_logits, wdl_logits, auxiliary_logits = outputs
    policy_targets = dataset.policy_targets[indices].to(device)
    wdl_targets = dataset.wdl_targets[indices].to(device)
    next_policy_targets = dataset.next_policy_targets[indices].to(device)
    remaining_targets = dataset.remaining_length_targets[indices].to(device)
    policy_loss = -(policy_targets * torch.log_softmax(policy_logits.float(), dim=1)).sum(dim=1).mean()
    wdl_loss = -(wdl_targets * torch.log_softmax(wdl_logits.float(), dim=1)).sum(dim=1).mean()
    next_policy_loss = -(next_policy_targets * torch.log_softmax(auxiliary_logits[0].float(), dim=1)).sum(dim=1).mean()
    remaining_loss = torch.nn.functional.smooth_l1_loss(auxiliary_logits[1].float(), remaining_targets)
    return policy_loss + wdl_loss + next_policy_loss + remaining_loss


def _train_step(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    dataset: FrozenReplayDataset,
    indices: Tensor,
    device: torch.device,
) -> None:
    states = dataset.states[indices].to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(states)
        loss = _training_loss(outputs, dataset, indices, device)
    loss.backward()
    optimizer.step()


def _training_measurement(
    model: DistributedDataParallel,
    dataset: FrozenReplayDataset,
    plan: ArchitectureBenchmarkPlan,
    protocol: ComparisonProtocol,
    rank: int,
    device: torch.device,
) -> TrainingMeasurement:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001, amsgrad=True, eps=1e-5)
    for step in range(plan.training.warmup_optimizer_steps):
        _train_step(model, optimizer, dataset, _batch_indices(dataset.sample_count, step, rank, plan), device)
    distributed.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    completed_steps = 0
    while True:
        _train_step(
            model,
            optimizer,
            dataset,
            _batch_indices(dataset.sample_count, completed_steps, rank, plan),
            device,
        )
        completed_steps += 1
        if protocol is ComparisonProtocol.EQUAL_SAMPLES:
            if completed_steps >= plan.training.equal_sample_optimizer_steps:
                break
        else:
            elapsed_tensor = torch.tensor(time.perf_counter() - started, device=device)
            distributed.all_reduce(elapsed_tensor, op=distributed.ReduceOp.MAX)
            if elapsed_tensor.item() >= plan.training.equal_wall_time_seconds:
                break
    torch.cuda.synchronize(device)
    distributed.barrier()
    elapsed = time.perf_counter() - started
    measurements = torch.tensor(
        (elapsed, torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)),
        dtype=torch.float64,
        device=device,
    )
    distributed.all_reduce(measurements, op=distributed.ReduceOp.MAX)
    elapsed = measurements[0].item()
    global_samples = completed_steps * plan.topology.global_training_batch_size
    return TrainingMeasurement(
        protocol=protocol,
        optimizer_steps=completed_steps,
        global_samples=global_samples,
        elapsed_seconds=elapsed,
        samples_per_second=global_samples / elapsed,
        maximum_rank_peak_allocated_bytes=int(measurements[1].item()),
        maximum_rank_peak_reserved_bytes=int(measurements[2].item()),
    )


def _inference_measurements(
    network: Network,
    plan: ArchitectureBenchmarkPlan,
    device: torch.device,
    world_size: int,
) -> tuple[InferenceMeasurement, ...]:
    network.eval()
    measurements: list[InferenceMeasurement] = []
    dimensions = network.dimensions
    with torch.inference_mode():
        for batch_size in plan.inference.batch_sizes:
            states = torch.zeros(
                batch_size,
                dimensions.channels,
                dimensions.rows,
                dimensions.columns,
                device=device,
            )
            for _ in range(plan.inference.warmup_batches):
                network(states)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for _ in range(plan.inference.measured_batches):
                network(states)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            mean_seconds = elapsed / plan.inference.measured_batches
            rank_measurements = torch.tensor(
                (
                    mean_seconds,
                    torch.cuda.max_memory_allocated(device),
                    torch.cuda.max_memory_reserved(device),
                ),
                dtype=torch.float64,
                device=device,
            )
            distributed.all_reduce(rank_measurements, op=distributed.ReduceOp.MAX)
            maximum_mean_seconds = rank_measurements[0].item()
            measurements.append(
                InferenceMeasurement(
                    batch_size=batch_size,
                    mean_seconds=maximum_mean_seconds,
                    minimum_rank_positions_per_second=batch_size / maximum_mean_seconds,
                    aggregate_positions_per_second=batch_size * world_size / maximum_mean_seconds,
                    maximum_rank_peak_allocated_bytes=int(rank_measurements[1].item()),
                    maximum_rank_peak_reserved_bytes=int(rank_measurements[2].item()),
                )
            )
    return tuple(measurements)


def _run_benchmark(arguments: argparse.Namespace, plan: ArchitectureBenchmarkPlan) -> None:
    if not arguments.acknowledge_gpu_load:
        raise ValueError('GPU benchmark requires --acknowledge-gpu-load.')
    rank, world_size, device = _distributed_context(plan)
    entry = _catalog_entry(plan, arguments.model_id)
    definition = entry.definition
    network = Network(
        definition.architecture,
        device,
        definition.dimensions,
        definition.auxiliary_output_sizes,
    )
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    if parameter_count != entry.expected_training_parameters:
        raise ValueError('Constructed parameter count disagrees with the architecture catalog.')
    model = DistributedDataParallel(TrainingNetwork(network), device_ids=[device.index])
    dataset = _load_frozen_replay(arguments.frozen_replay)
    training = _training_measurement(model, dataset, plan, arguments.protocol, rank, device)
    model.zero_grad(set_to_none=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    inference = _inference_measurements(network, plan, device, world_size)
    if rank == 0:
        result = ArchitectureBenchmarkResult(
            model_id=entry.model_id,
            parameter_count=parameter_count,
            world_size=world_size,
            plan=plan,
            frozen_replay_sha256=_sha256(arguments.frozen_replay),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda or 'none',
            device_name=torch.cuda.get_device_name(device),
            training=training,
            inference=inference,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(result.model_dump_json(indent=2) + '\n', encoding='utf-8')
    distributed.destroy_process_group()


def main() -> None:
    arguments = _parse_arguments()
    plan = load_architecture_benchmark_plan(arguments.plan)
    if arguments.command == 'describe':
        print(json.dumps(plan.model_dump(mode='json'), indent=2))
        return
    _run_benchmark(arguments, plan)


if __name__ == '__main__':
    main()
