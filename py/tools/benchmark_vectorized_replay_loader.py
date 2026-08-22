from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
from src.experiment.configuration import load_experiment_configuration
from src.games.composition import create_game_implementation
from src.replay.batch_loader import build_dense_targets, decode_states
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


class BenchmarkReport(FrozenModel):
    batch_size: int
    iterations: int
    rows: int
    index_generation_seconds: float
    index_generation_rows_per_second: float
    physical_mapping_seconds: float
    physical_mapping_rows_per_second: float
    gather_seconds: float
    gather_rows_per_second: float
    decode_seconds: float
    decode_rows_per_second: float
    augmentation_seconds: float
    augmentation_rows_per_second: float
    dense_target_seconds: float
    dense_target_rows_per_second: float
    pinned_fill_status: str = 'pending reusable-pinned-buffer phase'


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def _sample_plan(
    seed: tuple[int, int],
    replay_size: int,
    batch_size: int,
    augmentation_count: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    generator = np.random.default_rng(np.random.SeedSequence(seed))
    return (
        generator.choice(replay_size, size=batch_size, replace=False),
        generator.integers(0, augmentation_count, size=batch_size, dtype=np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark vectorized replay gathering and dense batch construction.')
    parser.add_argument('--configuration', required=True, type=Path)
    parser.add_argument('--replay-store', required=True, type=Path)
    parser.add_argument('--batch-size', default=256, type=_positive_int)
    parser.add_argument('--iterations', default=100, type=_positive_int)
    parser.add_argument('--seed', default=20260822, type=int)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()

    configuration = load_experiment_configuration(arguments.configuration)
    game = create_game_implementation(configuration)
    layout = ReplayLayout(
        packed_planes=game.state.packed_plane_layout,
        targets=game.target_layout,
        maximum_policy_entries=game.training.lifecycle.replay.maximum_policy_entries,
        maximum_legal_actions=game.state.maximum_legal_action_count,
    )
    try:
        store = ReplayStore.open(arguments.replay_store, layout, writable=False)
        try:
            plan_seeds = tuple((arguments.seed, iteration) for iteration in range(arguments.iterations))
            started = time.perf_counter()
            plans = tuple(
                _sample_plan(seed, store.state.size, arguments.batch_size, game.state.augmentation_count)
                for seed in plan_seeds
            )
            index_generation_seconds = time.perf_counter() - started

            started = time.perf_counter()
            physical_plans = tuple(store.logical_to_physical(logical) for logical, _ in plans)
            physical_mapping_seconds = time.perf_counter() - started

            started = time.perf_counter()
            gathered = tuple(store.gather_physical(physical) for physical in physical_plans)
            gather_seconds = time.perf_counter() - started

            started = time.perf_counter()
            decoded = tuple(decode_states(columns.encoded_state, game.state) for columns in gathered)
            decode_seconds = time.perf_counter() - started

            augmented = tuple(states.copy() for states in decoded)
            started = time.perf_counter()
            for states, (_, augmentation_indices) in zip(augmented, plans, strict=True):
                game.state.transform_decoded_states(states, augmentation_indices)
            augmentation_seconds = time.perf_counter() - started

            _ = game.state.action_permutations
            started = time.perf_counter()
            for columns, (_, augmentation_indices) in zip(gathered, plans, strict=True):
                build_dense_targets(columns, layout, game.state, augmentation_indices)
            dense_target_seconds = time.perf_counter() - started
        finally:
            store.close()
    finally:
        game.close()

    rows = arguments.batch_size * arguments.iterations
    report = BenchmarkReport(
        batch_size=arguments.batch_size,
        iterations=arguments.iterations,
        rows=rows,
        index_generation_seconds=index_generation_seconds,
        index_generation_rows_per_second=rows / index_generation_seconds,
        physical_mapping_seconds=physical_mapping_seconds,
        physical_mapping_rows_per_second=rows / physical_mapping_seconds,
        gather_seconds=gather_seconds,
        gather_rows_per_second=rows / gather_seconds,
        decode_seconds=decode_seconds,
        decode_rows_per_second=rows / decode_seconds,
        augmentation_seconds=augmentation_seconds,
        augmentation_rows_per_second=rows / augmentation_seconds,
        dense_target_seconds=dense_target_seconds,
        dense_target_rows_per_second=rows / dense_target_seconds,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomically(arguments.output, report.model_dump_json(indent=2) + '\n')


if __name__ == '__main__':
    main()
