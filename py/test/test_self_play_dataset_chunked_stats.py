from __future__ import annotations

import numpy as np
from pathlib import Path

from src.self_play.SelfPlayDataset import ReplaySampleMetadata, SelfPlayDataset
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats
from src.self_play.value_target import ReplayValueTarget, TerminationReason


def test_chunked_save_preserves_iteration_level_termination_stats(tmp_path: Path) -> None:
    dataset = SelfPlayDataset()
    dataset.encoded_states = [b'first', b'second', b'third']
    dataset.visit_counts = [np.array([[0, 1]], dtype=np.uint16) for _ in dataset.encoded_states]
    dataset.value_targets = [
        ReplayValueTarget.from_scores(0.0, 0.0, TerminationReason.NATURAL) for _ in dataset.encoded_states
    ]
    dataset.sample_metadata = [
        ReplaySampleMetadata(ply=ply, current_player_piece_count=8, opponent_piece_count=8)
        for ply in range(len(dataset.encoded_states))
    ]
    dataset.stats = SelfPlayDatasetStats(
        num_samples=3,
        num_games=2,
        game_lengths=[100, 250],
        total_generation_time=12.5,
        num_too_long_games=1,
        capped_game_material_scores=[0.25],
        low_material_termination_evaluations=2,
        low_material_terminations=1,
        low_material_termination_declines=1,
        low_material_termination_material_scores=[-0.5],
    )

    dataset.chunked_save(tmp_path, iteration=7, chunk_size=1)

    assert SelfPlayDataset.load_iteration_stats(tmp_path, iteration=7) == dataset.stats
