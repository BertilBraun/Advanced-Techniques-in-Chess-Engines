from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

from src.games.contracts import GameStateContract, Player, WdlTarget
from src.games.representation import PackedPlaneLayout, PackedPlanePayload, RepresentationDimensions
from src.replay.contracts import ReplaySample
from src.self_play.completed_game import TerminationReason


@dataclass(frozen=True)
class FakePosition:
    actions: tuple[int, ...]


@dataclass(frozen=True)
class FakeGameStateSpecification:
    name: str
    action_size: int
    terminal_ply: int
    representation: RepresentationDimensions
    legal_action_ids: tuple[int, ...]
    encode_position: Callable[[tuple[int, ...]], bytes]
    legal_actions_at_terminal: bool


class FakeGameState(GameStateContract[FakePosition]):
    def __init__(self, specification: FakeGameStateSpecification) -> None:
        self.specification = specification

    @property
    def name(self) -> str:
        return self.specification.name

    @property
    def action_size(self) -> int:
        return self.specification.action_size

    @property
    def representation(self) -> RepresentationDimensions:
        return self.specification.representation

    def initial_position(self) -> FakePosition:
        return FakePosition(())

    def _is_terminal(self, position: FakePosition) -> bool:
        return len(position.actions) >= self.specification.terminal_ply

    def legal_action_ids(self, position: FakePosition) -> tuple[int, ...]:
        if self._is_terminal(position) and not self.specification.legal_actions_at_terminal:
            return ()
        return self.specification.legal_action_ids

    def child_position(self, position: FakePosition, action_id: int) -> FakePosition:
        if action_id not in self.legal_action_ids(position):
            raise ValueError('Illegal fake action.')
        return FakePosition((*position.actions, action_id))

    def is_irreversible_transition(self, position: FakePosition, action_id: int, child: FakePosition) -> bool:
        del position, action_id, child
        return False

    def current_player(self, position: FakePosition) -> Player:
        return Player.FIRST if len(position.actions) % 2 == 0 else Player.SECOND

    def natural_terminal_wdl(self, position: FakePosition) -> WdlTarget | None:
        return WdlTarget(win=0.0, draw=1.0, loss=0.0) if self._is_terminal(position) else None

    def adjudicated_wdl(self, position: FakePosition, reason: TerminationReason) -> WdlTarget:
        del position, reason
        return WdlTarget(win=0.0, draw=1.0, loss=0.0)

    def encode_network_input(self, position: FakePosition) -> PackedPlanePayload:
        return self.packed_plane_layout.value(self.specification.encode_position(position.actions))

    @property
    def augmentation_count(self) -> int:
        return 1

    def transform_action_id(self, action_id: int, augmentation_index: int) -> int:
        del augmentation_index
        return action_id

    def transform_encoded_state(
        self,
        encoded_state: PackedPlanePayload,
        augmentation_index: int,
    ) -> PackedPlanePayload:
        del augmentation_index
        return encoded_state

    def transform_replay_targets(self, sample: ReplaySample, augmentation_index: int) -> ReplaySample:
        del augmentation_index
        return sample


def binary_choice_fake_game_state(terminal_ply: int = 6) -> FakeGameState:
    return FakeGameState(
        FakeGameStateSpecification(
            name='chess',
            action_size=2,
            terminal_ply=terminal_ply,
            representation=RepresentationDimensions(1, 1, 1, (), (0,), PackedPlaneLayout(1, 0, 1)),
            legal_action_ids=(0, 1),
            encode_position=lambda actions: bytes((len(actions),)),
            legal_actions_at_terminal=False,
        )
    )


def go_like_fake_game_state(
    *,
    action_size: int = 4,
    legal_action_ids: tuple[int, ...] = (0, 1, 2, 3),
    terminal_ply: int = 60,
    legal_actions_at_terminal: bool = False,
) -> FakeGameState:
    return FakeGameState(
        FakeGameStateSpecification(
            name='go',
            action_size=action_size,
            terminal_ply=terminal_ply,
            representation=RepresentationDimensions(1, 8, 8, (0,), (), PackedPlaneLayout(8, 1, 0)),
            legal_action_ids=legal_action_ids,
            encode_position=lambda actions: hashlib.sha256(bytes(actions)).digest()[:8],
            legal_actions_at_terminal=legal_actions_at_terminal,
        )
    )
