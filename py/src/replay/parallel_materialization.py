from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from pydantic import Field
from src.experiment.configuration import load_experiment_configuration_json
from src.games.composition import ConfiguredGame, create_game_implementation
from src.games.contracts import GameStateContract, TerminalOracle, WdlTarget
from src.replay.layout import ReplayLayout
from src.replay.materialization import materialize_completed_game
from src.replay.store import encode_replay_rows
from src.self_play.completed_game import (
    CompletedSelfPlayGame,
    GameIdentity,
    SearchObservation,
    TerminationReason,
)
from src.util.atomic_file import write_bytes_atomically, write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.generation_schedule import FloatGenerationSchedule

STAGED_ROWS_SUFFIX = '.rows.npy'
STAGED_METADATA_SUFFIX = '.meta.json'

PositionT = TypeVar('PositionT')


class StagedGameMetadata(FrozenModel):
    schema_version: Literal[1] = 1
    identity: GameIdentity
    row_count: int = Field(ge=0)
    length_plies: int = Field(ge=0)
    termination_reason: TerminationReason
    is_resignation_continuation: bool
    final_wdl: WdlTarget
    observations: tuple[SearchObservation, ...]
    policies_truncated: int = Field(ge=0)
    retained_visit_mass: int = Field(ge=0)
    discarded_visit_mass: int = Field(ge=0)


@dataclass(frozen=True)
class StagedGame:
    game_id: str
    row_count: int


def staged_rows_path(staging_path: Path, game_id: str) -> Path:
    return staging_path / f'{game_id}{STAGED_ROWS_SUFFIX}'


def staged_metadata_path(staging_path: Path, game_id: str) -> Path:
    return staging_path / f'{game_id}{STAGED_METADATA_SUFFIX}'


def stage_completed_game(
    inbox_file: Path,
    staging_path: Path,
    state: GameStateContract[PositionT],
    terminal_oracle: TerminalOracle[PositionT] | None,
    layout: ReplayLayout,
    value_discount_per_ply: FloatGenerationSchedule,
    censor_remaining_game_length_on_cut_games: bool,
) -> StagedGame:
    game = CompletedSelfPlayGame.model_validate_json(inbox_file.read_text(encoding='utf-8'))
    if inbox_file.name != game.identity.file_name:
        raise ValueError(f'Completed-game identity does not match its file name: {inbox_file}')
    materialized = materialize_completed_game(
        game,
        state,
        terminal_oracle,
        layout.targets,
        layout.maximum_policy_entries,
        value_discount_per_ply,
        censor_remaining_game_length_on_cut_games=censor_remaining_game_length_on_cut_games,
    )
    rows = encode_replay_rows(layout, materialized.samples)
    metadata = StagedGameMetadata(
        identity=game.identity,
        row_count=len(rows),
        length_plies=len(game.action_ids),
        termination_reason=game.termination_reason,
        is_resignation_continuation=game.is_resignation_continuation,
        final_wdl=game.final_wdl,
        observations=game.observations,
        policies_truncated=materialized.policies_truncated,
        retained_visit_mass=materialized.retained_visit_mass,
        discarded_visit_mass=materialized.discarded_visit_mass,
    )
    game_id = inbox_file.stem
    buffer = io.BytesIO()
    np.save(buffer, rows, allow_pickle=False)
    write_bytes_atomically(staged_rows_path(staging_path, game_id), buffer.getvalue())
    # Metadata is written last: its presence marks the staged game as complete.
    write_text_atomically(staged_metadata_path(staging_path, game_id), metadata.model_dump_json() + '\n')
    inbox_file.unlink(missing_ok=True)
    return StagedGame(game_id=game_id, row_count=len(rows))


_worker_game: ConfiguredGame | None = None
_worker_layout: ReplayLayout | None = None


def initialize_materialization_worker(configuration_json: str, maximum_policy_entries: int) -> None:
    global _worker_game, _worker_layout
    _worker_game = create_game_implementation(load_experiment_configuration_json(configuration_json))
    _worker_layout = ReplayLayout(
        packed_planes=_worker_game.state.packed_plane_layout,
        targets=_worker_game.target_layout,
        maximum_policy_entries=maximum_policy_entries,
        maximum_legal_actions=_worker_game.state.maximum_legal_action_count,
    )


def stage_completed_game_path(inbox_file: Path, staging_path: Path) -> StagedGame:
    if _worker_game is None or _worker_layout is None:
        raise RuntimeError('Replay materialization worker has not been initialized.')
    return stage_completed_game(
        inbox_file,
        staging_path,
        _worker_game.state,
        _worker_game.terminal_oracle,
        _worker_layout,
        _worker_game.value_discount_per_ply,
        _worker_game.censor_remaining_game_length_on_cut_games,
    )
