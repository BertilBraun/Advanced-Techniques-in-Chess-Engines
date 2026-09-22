from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from src.experiment.configuration import load_experiment_configuration
from src.games.composition import create_game_implementation
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore


def _store_layout(configuration_path: Path) -> tuple[ReplayLayout, Path]:
    experiment = load_experiment_configuration(configuration_path)
    game = create_game_implementation(experiment)
    layout = ReplayLayout(
        packed_planes=game.state.packed_plane_layout,
        targets=game.target_layout,
        maximum_policy_entries=experiment.training.lifecycle.replay.maximum_policy_entries,
        maximum_legal_actions=game.state.maximum_legal_action_count,
    )
    return layout, Path(experiment.training.save_path) / 'replay.bin'


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Drop the newest replay rows produced by a model that should not have been serving.'
    )
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--first-contaminated-generation', type=int, required=True)
    # The configuration's save path is repository relative, so a run inspected from another
    # checkout has to name its store directly.
    parser.add_argument('--replay-path', type=Path)
    # Self-play finishes asynchronously, so games from before the boundary keep arriving after it.
    # Truncating a FIFO cannot skip them: removing every contaminated row also removes the clean
    # rows appended after the first of them.
    parser.add_argument('--accept-collateral', action='store_true')
    parser.add_argument('--apply', action='store_true')
    arguments = parser.parse_args()

    layout, resolved_path = _store_layout(arguments.configuration)
    replay_path = arguments.replay_path or resolved_path
    store = ReplayStore.open(replay_path, layout, writable=arguments.apply)
    try:
        state = store.state
        generations = store.logical_source_model_generations()
        contaminated = generations >= arguments.first_contaminated_generation
        print(f'window          : {state.size} rows, head {state.head}, appended {state.total_appended_rows}')
        print(f'generations     : {int(generations.min())} to {int(generations.max())}')
        print(f'contaminated    : {int(contaminated.sum())} rows at or after {arguments.first_contaminated_generation}')
        if not contaminated.any():
            print('nothing to prune')
            return
        first = int(np.argmax(contaminated))
        # The store is a FIFO, so anything a model produced sits at the end. Pruning is only ever a
        # truncation of the newest rows; a hole in the middle would mean the window is not what we think.
        clean_after = int((~contaminated[first:]).sum())
        if clean_after and not arguments.accept_collateral:
            raise SystemExit(
                f'refusing to prune: {clean_after} clean rows sit after the first contaminated row; '
                'pass --accept-collateral to remove them too'
            )
        removed = state.size - first
        if clean_after:
            print(f'collateral      : {clean_after} clean rows removed alongside the contaminated ones')
        print(f'would keep      : {first} rows (generations up to {int(generations[first - 1])})')
        print(f'would remove    : {removed} rows ({removed / state.size:.1%} of the window)')
        if arguments.apply:
            store.truncate_newest(removed)
            print(f'REMOVED; window is now {store.state.size} rows')
    finally:
        store.close()


if __name__ == '__main__':
    main()
