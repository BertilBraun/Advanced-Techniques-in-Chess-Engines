from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import psutil
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import default_collate

from src.cluster.TrainerProcess import as_dataloader
from src.experiment.run_configuration import apply_run_configuration, load_run_configuration
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.settings import TRAINING_ARGS
from src.train.Trainer import Trainer, prefetch_training_batches
from src.util.save_paths import load_model_and_optimizer


@dataclass(frozen=True)
class ThroughputResult:
    samples: int
    batches: int
    dataloader_workers: int
    legacy_decode_seconds_per_batch: float
    vectorized_decode_seconds_per_batch: float
    loader_seconds_per_batch: float
    cuda_seconds_per_batch: float
    total_seconds_per_batch: float
    estimated_gpu_duty_cycle_percent: float
    host_ram_gib: float
    gpu_memory_mib: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-config', required=True, type=Path)
    parser.add_argument('--iteration', required=True, type=int)
    parser.add_argument('--batches', default=30, type=int)
    parser.add_argument('--warmup-batches', default=3, type=int)
    parser.add_argument('--dataloader-workers', default=0, type=int)
    return parser.parse_args()


def load_iteration_dataset(save_path: str, iteration: int) -> SelfPlayDataset:
    dataset = SelfPlayDataset.load_iteration(save_path, iteration)
    if len(dataset) < TRAINING_ARGS.training.batch_size:
        raise ValueError(f'Iteration {iteration} contains only {len(dataset)} samples.')
    return dataset


def benchmark_decode(dataset: SelfPlayDataset, batches: int) -> tuple[float, float]:
    batch_size = TRAINING_ARGS.training.batch_size
    indices = list(range(batch_size))

    started_at = perf_counter()
    for _ in range(batches):
        default_collate([dataset[index] for index in indices])
    legacy_seconds = perf_counter() - started_at

    started_at = perf_counter()
    for _ in range(batches):
        dataset.__getitems__(indices)
    vectorized_seconds = perf_counter() - started_at
    return legacy_seconds / batches, vectorized_seconds / batches


def train_one_batch(
    trainer: Trainer,
    scaler: GradScaler,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    trainer.optimizer.zero_grad()
    with autocast(trainer.model.device.type, dtype=torch.bfloat16):
        loss = trainer._calculate_loss_for_batch(batch).total_loss
    scaler.scale(loss).backward()
    scaler.unscale_(trainer.optimizer)
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.args.max_grad_norm)
    scaler.step(trainer.optimizer)
    scaler.update()


def benchmark_training(
    dataset: SelfPlayDataset,
    iteration: int,
    warmup_batches: int,
    batches: int,
    dataloader_workers: int,
) -> tuple[float, float, float]:
    device = torch.device('cuda', TRAINING_ARGS.cluster.trainer_device_id)
    torch.cuda.set_device(device)
    model, optimizer = load_model_and_optimizer(
        iteration,
        TRAINING_ARGS.network,
        device,
        TRAINING_ARGS.save_path,
        TRAINING_ARGS.training.optimizer,
    )
    trainer = Trainer(model, optimizer, TRAINING_ARGS.training, (device.index,))
    scaler = GradScaler()
    dataloader = as_dataloader(
        dataset,
        TRAINING_ARGS.training.batch_size,
        num_workers=dataloader_workers,
        drop_last=True,
    )
    iterator = iter(prefetch_training_batches(dataloader))

    for _ in range(warmup_batches):
        train_one_batch(trainer, scaler, next(iterator))
    torch.cuda.synchronize(device)

    loader_seconds = 0.0
    cuda_seconds = 0.0
    total_started_at = perf_counter()
    for _ in range(batches):
        loader_started_at = perf_counter()
        batch = next(iterator)
        loader_seconds += perf_counter() - loader_started_at

        cuda_started_at = torch.cuda.Event(enable_timing=True)
        cuda_finished_at = torch.cuda.Event(enable_timing=True)
        cuda_started_at.record()
        train_one_batch(trainer, scaler, batch)
        cuda_finished_at.record()
        torch.cuda.synchronize(device)
        cuda_seconds += cuda_started_at.elapsed_time(cuda_finished_at) / 1000

    total_seconds = perf_counter() - total_started_at
    return loader_seconds / batches, cuda_seconds / batches, total_seconds / batches


def main() -> None:
    arguments = parse_arguments()
    configuration = load_run_configuration(arguments.run_config)
    apply_run_configuration(TRAINING_ARGS, configuration)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = load_iteration_dataset(TRAINING_ARGS.save_path, arguments.iteration)
    legacy_decode_seconds, vectorized_decode_seconds = benchmark_decode(dataset, arguments.batches)
    loader_seconds, cuda_seconds, total_seconds = benchmark_training(
        dataset,
        arguments.iteration,
        arguments.warmup_batches,
        arguments.batches,
        arguments.dataloader_workers,
    )
    result = ThroughputResult(
        samples=len(dataset),
        batches=arguments.batches,
        dataloader_workers=arguments.dataloader_workers,
        legacy_decode_seconds_per_batch=legacy_decode_seconds,
        vectorized_decode_seconds_per_batch=vectorized_decode_seconds,
        loader_seconds_per_batch=loader_seconds,
        cuda_seconds_per_batch=cuda_seconds,
        total_seconds_per_batch=total_seconds,
        estimated_gpu_duty_cycle_percent=100 * cuda_seconds / total_seconds,
        host_ram_gib=psutil.Process().memory_info().rss / 2**30,
        gpu_memory_mib=torch.cuda.memory_allocated(TRAINING_ARGS.cluster.trainer_device_id) / 2**20,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == '__main__':
    main()
