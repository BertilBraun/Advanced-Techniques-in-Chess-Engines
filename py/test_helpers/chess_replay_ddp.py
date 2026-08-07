from __future__ import annotations

from multiprocessing.connection import Connection
import sys
from types import ModuleType

import torch
import torch.distributed as distributed

sys.modules.setdefault('GPUtil', ModuleType('GPUtil'))

from src.cluster.TrainerProcess import MaintainReplayCommand, _maintain_replay
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.self_play.value_target import ReplayValueTarget, TerminationReason
from src.games.chess.replay import pack_chess_visits
from src.train.Replay import PackedReplaySample, ReplayMetrics, ReplaySnapshot
from src.train.training_batch import ReplaySampleMetadata


WORLD_SIZE = 2


class SnapshotMaintainer:
    def __init__(self, snapshot: ReplaySnapshot, metrics: ReplayMetrics) -> None:
        self.snapshot = snapshot
        self.metrics = metrics

    def maintain(self, capacity: int) -> tuple[ReplaySnapshot, ReplayMetrics]:
        assert capacity == len(self.snapshot.samples)
        return self.snapshot, self.metrics


def replay_snapshot() -> tuple[ReplaySnapshot, ReplayMetrics]:
    sample = PackedReplaySample(
        encoded_state=CHESS_STATE_CONTRACT.representation.packed_planes.empty_value(),
        visits=pack_chess_visits(((0, 1),)),
        value_target=ReplayValueTarget.from_scores(0.0, 0.0, TerminationReason.NATURAL),
        metadata=ReplaySampleMetadata(ply=0, current_player_piece_count=16, opponent_piece_count=16),
        sample_weight=1.0,
        source_model_generation=0,
        source_created_at_seconds=1.0,
    )
    samples = (sample,) * 8
    snapshot = ReplaySnapshot(
        samples=samples,
        credited_samples=12,
        credited_completed_searches=96,
        sampler_seed=71,
        frozen_at_seconds=2.0,
        evicted_samples=4,
        estimated_sample_bytes=1_024,
        encoded_state_value_overhead_bytes=0,
        projected_capacity_bytes=1_024,
        projected_review_capacity_bytes=1_024,
    )
    metrics = ReplayMetrics(
        credited_samples=12,
        credited_completed_searches=96,
        live_samples=8,
        evicted_samples=4,
        oldest_source_model_generation=0,
        newest_source_model_generation=0,
        mean_source_model_generation=0.0,
        oldest_sample_age_seconds=1.0,
        mean_sample_age_seconds=1.0,
        estimated_sample_bytes=1_024,
        encoded_state_value_overhead_bytes=0,
        projected_capacity_bytes=1_024,
        projected_review_capacity_bytes=1_024,
    )
    return snapshot, metrics


def run_replay_rank(
    rank: int,
    initialization_method: str,
    connection: Connection,
) -> None:
    distributed.init_process_group(
        backend='gloo',
        init_method=initialization_method,
        rank=rank,
        world_size=WORLD_SIZE,
    )
    try:
        snapshot, metrics = replay_snapshot()
        maintainer = SnapshotMaintainer(snapshot, metrics) if rank == 0 else None
        response, received = _maintain_replay(
            MaintainReplayCommand(phase_id=1, replay_capacity_unique_positions=8),
            maintainer,
            rank,
            torch.device('cpu'),
        )
        indices = received.rank_indices(0, 1, 4, WORLD_SIZE, rank)
        connection.send(
            (
                response.credited_unique_samples,
                response.live_unique_samples,
                response.evicted_unique_samples,
                response.replay_memory_bytes,
                indices,
            )
        )
    finally:
        distributed.destroy_process_group()
        connection.close()
