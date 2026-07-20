from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import torch
from torch.amp import GradScaler, autocast

from src.Network import Network
from src.settings import CurrentGame, TRAINING_ARGS
from src.train.Trainer import Trainer
from src.train.TrainingArgs import NetworkParams, SEPlacement


@dataclass(frozen=True)
class Arguments:
    device: int
    layers: int
    hidden_size: int
    se_placement: SEPlacement
    batches: int
    warmup_batches: int


@dataclass(frozen=True)
class Result:
    parameters: int
    seconds_per_batch: float
    peak_gpu_memory_mib: float


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True, type=int)
    parser.add_argument('--layers', required=True, type=int)
    parser.add_argument('--hidden-size', required=True, type=int)
    parser.add_argument(
        '--se-placement',
        choices=tuple(SEPlacement),
        default=SEPlacement.EVERY_SECOND_BLOCK,
        type=SEPlacement,
    )
    parser.add_argument('--batches', default=5, type=int)
    parser.add_argument('--warmup-batches', default=2, type=int)
    namespace = parser.parse_args()
    return Arguments(
        device=namespace.device,
        layers=namespace.layers,
        hidden_size=namespace.hidden_size,
        se_placement=namespace.se_placement,
        batches=namespace.batches,
        warmup_batches=namespace.warmup_batches,
    )


def train_batch(
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


def main() -> None:
    arguments = parse_arguments()
    device = torch.device('cuda', arguments.device)
    torch.cuda.set_device(device)
    model = Network(
        NetworkParams(
            arguments.layers,
            arguments.hidden_size,
            arguments.se_placement,
        ),
        device,
    )
    optimizer = torch.optim.AdamW(model.parameters())
    trainer = Trainer(model, optimizer, TRAINING_ARGS.training)
    scaler = GradScaler()

    batch_size = TRAINING_ARGS.training.batch_size
    state = torch.randn((batch_size, *CurrentGame.representation_shape))
    policy = torch.zeros((batch_size, CurrentGame.action_size))
    policy[:, 0] = 1
    value = torch.zeros(batch_size)
    batch = (state, policy, value)

    for _ in range(arguments.warmup_batches):
        train_batch(trainer, scaler, batch)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    started_at = time.perf_counter()
    for _ in range(arguments.batches):
        train_batch(trainer, scaler, batch)
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at

    result = Result(
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        seconds_per_batch=elapsed_seconds / arguments.batches,
        peak_gpu_memory_mib=torch.cuda.max_memory_allocated(device) / 2**20,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == '__main__':
    main()
