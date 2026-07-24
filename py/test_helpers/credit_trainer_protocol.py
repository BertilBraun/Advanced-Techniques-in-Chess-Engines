from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path
from types import ModuleType

import torch
import torch.distributed as distributed

sys.modules.setdefault('GPUtil', ModuleType('GPUtil'))

from src.cluster.CreditTrainerProcess import (
    MaintainCreditReplayCommand,
    TrainCreditQuantumCommand,
)
from src.train.RollingReplayBuffer import ReplayQuantumRequest, RollingReplayBuffer, decode_rank_quantum


WORLD_SIZE = 4


def run_gloo_protocol_rank(
    rank: int,
    initialization_method: str,
    output_queue: multiprocessing.Queue[tuple[int, tuple[tuple[int, ...], ...]]],
) -> None:
    distributed.init_process_group(
        backend='gloo',
        init_method=initialization_method,
        rank=rank,
        world_size=WORLD_SIZE,
    )
    try:
        commands = (
            MaintainCreditReplayCommand(
                phase_id=1,
                replay_capacity_unique_positions=100_000,
                compact_below_credited_unique_samples=None,
            ),
            TrainCreditQuantumCommand(phase_id=2, global_step=1_250, model_version=26),
        )
        observed: list[tuple[int, ...]] = []
        for command in commands:
            match command:
                case MaintainCreditReplayCommand():
                    local = torch.tensor(
                        [command.phase_id, 0, 0, rank],
                        dtype=torch.int64,
                    )
                case TrainCreditQuantumCommand():
                    local = torch.tensor(
                        [
                            command.phase_id,
                            command.global_step,
                            command.model_version,
                            rank,
                        ],
                        dtype=torch.int64,
                    )
            gathered = [torch.zeros_like(local) for _ in range(WORLD_SIZE)]
            distributed.all_gather(gathered, local)
            observed.extend(tuple(int(value) for value in values.tolist()) for values in gathered)
            distributed.barrier()
        output_queue.put((rank, tuple(observed)))
    finally:
        distributed.destroy_process_group()


def run_gloo_decode_rank(
    rank: int,
    initialization_method: str,
    replay_inbox: str,
    index_path: str,
    output_queue: multiprocessing.Queue[tuple[int, tuple[int, ...], int]],
) -> None:
    distributed.init_process_group(
        backend='gloo',
        init_method=initialization_method,
        rank=rank,
        world_size=WORLD_SIZE,
    )
    try:
        replay = RollingReplayBuffer(
            replay_inbox=Path(replay_inbox),
            index_path=Path(index_path),
            sampler_seed=37,
            read_only=True,
        )
        quantum = decode_rank_quantum(
            replay,
            ReplayQuantumRequest(
                global_step=10,
                optimizer_steps=2,
                global_batch_size=4,
                world_size=WORLD_SIZE,
                rank=rank,
            ),
        )
        distributed.barrier()
        output_queue.put(
            (
                rank,
                tuple(int(ply) for ply in quantum.full_batch.plies.tolist()),
                quantum.optimizer_steps,
            )
        )
    finally:
        distributed.destroy_process_group()
