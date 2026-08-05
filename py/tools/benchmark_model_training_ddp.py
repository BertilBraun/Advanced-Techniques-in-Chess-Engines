from __future__ import annotations

import argparse
from collections.abc import Iterator
import json
import socket
import time
from dataclasses import asdict, dataclass
from enum import StrEnum

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
import torch.nn.functional as functional
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from src.Network import Network
from src.self_play.SelfPlayDataset import TrainingBatch
from src.self_play.value_target import FinalOutcome, TerminationReason
from src.settings import CurrentGame, TRAINING_ARGS
from src.train.Trainer import Trainer
from src.train.TrainingArgs import NetworkParams, SEPlacement, TrainingParams
from src.value import scalar_to_wdl


class TrainingPath(StrEnum):
    MINIMAL = 'minimal'
    PRODUCTION = 'production'


@dataclass(frozen=True)
class Arguments:
    device_ids: tuple[int, ...]
    local_batch_size: int
    layers: int
    hidden_size: int
    se_placement: SEPlacement
    batches: int
    warmup_batches: int
    training_path: TrainingPath


@dataclass(frozen=True)
class Result:
    parameters: int
    device_ids: tuple[int, ...]
    local_batch_size: int
    global_batch_size: int
    seconds_per_batch: float
    samples_per_second: float
    peak_gpu_memory_mib: tuple[float, ...]
    training_path: TrainingPath


@dataclass(frozen=True)
class FixedBatchLoader:
    batch: TrainingBatch
    repetitions: int

    def __iter__(self) -> Iterator[TrainingBatch]:
        for _ in range(self.repetitions):
            yield self.batch

    def __len__(self) -> int:
        return self.repetitions


class LogitNetwork(nn.Module):
    def __init__(self, network: Network) -> None:
        super().__init__()
        self.network = network

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.network.logit_forward(state)


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-ids', required=True, nargs='+', type=int)
    parser.add_argument('--local-batch-size', required=True, type=int)
    parser.add_argument('--layers', required=True, type=int)
    parser.add_argument('--hidden-size', required=True, type=int)
    parser.add_argument(
        '--se-placement',
        choices=tuple(SEPlacement),
        default=SEPlacement.EVERY_SECOND_BLOCK,
        type=SEPlacement,
    )
    parser.add_argument('--batches', default=20, type=int)
    parser.add_argument('--warmup-batches', default=5, type=int)
    parser.add_argument(
        '--training-path',
        choices=tuple(TrainingPath),
        default=TrainingPath.MINIMAL,
        type=TrainingPath,
    )
    namespace = parser.parse_args()
    return Arguments(
        device_ids=tuple(namespace.device_ids),
        local_batch_size=namespace.local_batch_size,
        layers=namespace.layers,
        hidden_size=namespace.hidden_size,
        se_placement=namespace.se_placement,
        batches=namespace.batches,
        warmup_batches=namespace.warmup_batches,
        training_path=namespace.training_path,
    )


def available_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(('127.0.0.1', 0))
        return int(server.getsockname()[1])


def train_batch(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    batch: TrainingBatch,
) -> None:
    optimizer.zero_grad()
    with autocast('cuda', dtype=torch.bfloat16):
        policy_logits, value_logits = model(batch.states)
        policy_loss = functional.cross_entropy(policy_logits, batch.policy_targets)
        target_expected_scores = batch.final_outcomes.eq(int(FinalOutcome.WIN)).to(
            dtype=value_logits.dtype
        ) - batch.final_outcomes.eq(int(FinalOutcome.LOSS)).to(dtype=value_logits.dtype)
        value_loss = functional.cross_entropy(value_logits, scalar_to_wdl(target_expected_scores))
        total_loss = policy_loss + 0.5 * value_loss
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    scaler.step(optimizer)
    scaler.update()


def production_training_parameters(global_batch_size: int, local_batch_size: int) -> TrainingParams:
    return TrainingParams(
        num_epochs=1,
        global_batch_size=global_batch_size,
        local_batch_size=local_batch_size,
        optimizer='adamw',
        learning_rate=lambda _iteration, _optimizer: 0.0035,
        learning_rate_scheduler=lambda _progress, learning_rate: learning_rate,
        credit_training=TRAINING_ARGS.training.credit_training,
        outcome_value_loss_weight=0.85,
        mcts_value_loss_weight=0.15,
        mcts_value_loss_scale=1.0,
        mcts_value_target_warmup_optimizer_steps=50_000,
    )


def benchmark_rank(
    rank: int,
    arguments: Arguments,
    initialization_method: str,
) -> None:
    world_size = len(arguments.device_ids)
    device_id = arguments.device_ids[rank]
    torch.cuda.set_device(device_id)
    device = torch.device('cuda', device_id)
    distributed.init_process_group(
        backend='nccl',
        init_method=initialization_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(20260717)
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True

        network = Network(
            NetworkParams(
                arguments.layers,
                arguments.hidden_size,
                arguments.se_placement,
            ),
            device,
        )
        model = DistributedDataParallel(
            LogitNetwork(network),
            device_ids=[device_id],
            output_device=device_id,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = GradScaler()

        state = torch.randn(
            (arguments.local_batch_size, *CurrentGame.representation_shape),
            device=device,
        )
        policy_targets = torch.zeros(
            (arguments.local_batch_size, CurrentGame.action_size),
            device=device,
        )
        policy_targets[:, 0] = 1
        final_outcomes = torch.arange(arguments.local_batch_size, device=device, dtype=torch.int64) % len(FinalOutcome)
        batch = TrainingBatch(
            states=state,
            policy_targets=policy_targets,
            final_outcomes=final_outcomes,
            mcts_root_values=torch.zeros(arguments.local_batch_size, device=device),
            outcome_target_eligible=torch.ones(
                arguments.local_batch_size,
                dtype=torch.bool,
                device=device,
            ),
            material_result_scores=torch.zeros(arguments.local_batch_size, device=device),
            material_target_eligible=torch.zeros(
                arguments.local_batch_size,
                dtype=torch.bool,
                device=device,
            ),
            termination_reasons=torch.full(
                (arguments.local_batch_size,),
                int(TerminationReason.NATURAL),
                dtype=torch.int64,
                device=device,
            ),
            plies=torch.zeros(arguments.local_batch_size, dtype=torch.int32, device=device),
            current_player_piece_counts=torch.full(
                (arguments.local_batch_size,),
                8,
                dtype=torch.int8,
                device=device,
            ),
            opponent_piece_counts=torch.full(
                (arguments.local_batch_size,),
                8,
                dtype=torch.int8,
                device=device,
            ),
            occurrence_counts=torch.ones(arguments.local_batch_size, dtype=torch.int32, device=device),
        )

        global_batch_size = arguments.local_batch_size * world_size
        production_trainer = Trainer(
            network,
            optimizer,
            production_training_parameters(global_batch_size, arguments.local_batch_size),
            training_model=model,
            rank=rank,
        )
        match arguments.training_path:
            case TrainingPath.MINIMAL:
                for _ in range(arguments.warmup_batches):
                    train_batch(model, optimizer, scaler, batch)
            case TrainingPath.PRODUCTION:
                production_trainer.train(
                    FixedBatchLoader(batch, arguments.warmup_batches),
                    iteration=0,
                )
        torch.cuda.synchronize(device)
        distributed.barrier()
        torch.cuda.reset_peak_memory_stats(device)

        started_at = time.perf_counter()
        match arguments.training_path:
            case TrainingPath.MINIMAL:
                for _ in range(arguments.batches):
                    train_batch(model, optimizer, scaler, batch)
            case TrainingPath.PRODUCTION:
                production_trainer.train(
                    FixedBatchLoader(batch, arguments.batches),
                    iteration=0,
                )
        torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - started_at

        elapsed = torch.tensor(elapsed_seconds, device=device)
        distributed.all_reduce(elapsed, op=distributed.ReduceOp.MAX)
        peak_memory = torch.tensor(
            torch.cuda.max_memory_allocated(device) / 2**20,
            device=device,
        )
        gathered_peak_memory = [torch.zeros_like(peak_memory) for _ in range(world_size)]
        distributed.all_gather(gathered_peak_memory, peak_memory)

        if rank == 0:
            synchronized_elapsed_seconds = float(elapsed.item())
            result = Result(
                parameters=sum(parameter.numel() for parameter in network.parameters()),
                device_ids=arguments.device_ids,
                local_batch_size=arguments.local_batch_size,
                global_batch_size=global_batch_size,
                seconds_per_batch=synchronized_elapsed_seconds / arguments.batches,
                samples_per_second=global_batch_size * arguments.batches / synchronized_elapsed_seconds,
                peak_gpu_memory_mib=tuple(float(memory.item()) for memory in gathered_peak_memory),
                training_path=arguments.training_path,
            )
            print(json.dumps(asdict(result), indent=2), flush=True)
    finally:
        distributed.destroy_process_group()


def main() -> None:
    arguments = parse_arguments()
    if len(set(arguments.device_ids)) != len(arguments.device_ids):
        raise ValueError('DDP device IDs must be unique.')
    initialization_method = f'tcp://127.0.0.1:{available_tcp_port()}'
    multiprocessing.spawn(
        benchmark_rank,
        args=(arguments, initialization_method),
        nprocs=len(arguments.device_ids),
        join=True,
    )


if __name__ == '__main__':
    main()
