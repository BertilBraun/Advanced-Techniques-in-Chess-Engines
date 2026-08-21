from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.experiment.configuration import load_experiment_configuration_json
from src.games.composition import ConfiguredGame, create_game_implementation
from src.replay.materialization import MaterializedGame, materialize_completed_game
from src.self_play.completed_game import CompletedSelfPlayGame


@dataclass(frozen=True)
class MaterializedCompletedGame:
    completed_game: CompletedSelfPlayGame
    materialized_game: MaterializedGame


_worker_game: ConfiguredGame | None = None
_maximum_policy_entries: int | None = None


def initialize_materialization_worker(configuration_json: str, maximum_policy_entries: int) -> None:
    global _maximum_policy_entries, _worker_game
    _worker_game = create_game_implementation(load_experiment_configuration_json(configuration_json))
    _maximum_policy_entries = maximum_policy_entries


def materialize_completed_game_path(path: Path) -> MaterializedCompletedGame:
    if _worker_game is None or _maximum_policy_entries is None:
        raise RuntimeError('Replay materialization worker has not been initialized.')
    completed_game = CompletedSelfPlayGame.model_validate_json(path.read_text(encoding='utf-8'))
    if path.name != completed_game.identity.file_name:
        raise ValueError(f'Completed-game identity does not match its file name: {path}')
    return MaterializedCompletedGame(
        completed_game=completed_game,
        materialized_game=materialize_completed_game(
            completed_game,
            _worker_game.state,
            _worker_game.terminal_oracle,
            _worker_game.target_layout,
            _maximum_policy_entries,
            _worker_game.value_discount_per_ply,
            censor_remaining_game_length_on_cut_games=_worker_game.censor_remaining_game_length_on_cut_games,
        ),
    )
