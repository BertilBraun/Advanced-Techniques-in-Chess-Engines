from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

import numpy as np
import numpy.typing as npt
from AlphaZeroCpp import GameSearchVisit
from src.games.contracts import GameStateContract, Player, WdlTarget
from src.games.representation import (
    NetworkDimensions,
    PackedPlaneLayout,
    PackedPlanePayload,
    RepresentationDimensions,
    decode_packed_planes,
    decode_packed_planes_into,
    encode_packed_planes,
)
from src.replay.contracts import (
    EligibleLegalMovesTarget,
    EligibleNextPolicyTarget,
    EligibleRemainingGameLengthTarget,
    EligibleScalarAuxiliaryTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.self_play.completed_game import TerminationReason


class NativeEnumValue(Protocol):
    @property
    def name(self) -> str: ...


class NativeGoAreaScore(Protocol):
    @property
    def black_half_points(self) -> int: ...

    @property
    def white_half_points(self) -> int: ...

    @property
    def winner(self) -> NativeEnumValue | None: ...


class NativeGoPosition(Protocol):
    @property
    def board_size(self) -> int: ...

    @property
    def player(self) -> NativeEnumValue: ...

    @property
    def is_terminal(self) -> bool: ...

    @property
    def termination_reason(self) -> NativeEnumValue: ...

    def legal_actions(self) -> list[int]: ...

    def child(self, action_id: int) -> NativeGoPosition: ...

    def terminal_value(self) -> float: ...

    def area_score(self) -> NativeGoAreaScore: ...

    def packed_encoding(self) -> bytes: ...


class GoSymmetryIndex(IntEnum):
    IDENTITY = 0
    ROTATE_90 = 1
    ROTATE_180 = 2
    ROTATE_270 = 3
    REFLECT = 4
    REFLECT_ROTATE_90 = 5
    REFLECT_ROTATE_180 = 6
    REFLECT_ROTATE_270 = 7


@dataclass(frozen=True)
class GoStateContract(GameStateContract[NativeGoPosition]):
    board_size: int
    history_length: int = 8
    komi_half_points: int = 0
    maximum_moves: int | None = None

    def __post_init__(self) -> None:
        if self.board_size not in (7, 9):
            raise ValueError('Go state contract supports only 7x7 and 9x9 boards.')
        if self.history_length != 8:
            raise ValueError('Go state contract supports exactly eight history positions.')

    @property
    def binary_channels(self) -> tuple[int, ...]:
        return tuple(range(self.history_length * 2))

    @property
    def scalar_channels(self) -> tuple[int, ...]:
        return (self.history_length * 2,)

    @property
    def channels(self) -> int:
        return len(self.binary_channels) + len(self.scalar_channels)

    @property
    def action_size(self) -> int:
        return self.board_size**2 + 1

    @property
    def pass_action(self) -> int:
        return self.board_size**2

    @property
    def packed_planes(self) -> PackedPlaneLayout:
        return PackedPlaneLayout(self.board_size, len(self.binary_channels), len(self.scalar_channels))

    @property
    def network_dimensions(self) -> NetworkDimensions:
        return NetworkDimensions(self.channels, self.board_size, self.board_size, self.action_size)

    @property
    def name(self) -> str:
        return 'go'

    @property
    def representation(self) -> RepresentationDimensions:
        return RepresentationDimensions(
            channels=self.channels,
            rows=self.board_size,
            columns=self.board_size,
            binary_channels=self.binary_channels,
            scalar_channels=self.scalar_channels,
            packed_planes=self.packed_planes,
        )

    def initial_position(self) -> NativeGoPosition:
        from AlphaZeroCpp import GoPosition7, GoPosition9, GoRules

        maximum_moves = self.board_size * self.board_size * 4 if self.maximum_moves is None else self.maximum_moves
        rules = GoRules(self.komi_half_points, maximum_moves)
        return GoPosition7(rules) if self.board_size == 7 else GoPosition9(rules)

    def legal_action_ids(self, position: NativeGoPosition) -> tuple[int, ...]:
        return tuple(sorted(position.legal_actions()))

    def child_position(self, position: NativeGoPosition, action_id: int) -> NativeGoPosition:
        return position.child(action_id)

    def is_irreversible_transition(
        self,
        position: NativeGoPosition,
        action_id: int,
        child: NativeGoPosition,
    ) -> bool:
        del position, child
        return action_id != self.pass_action

    def current_player(self, position: NativeGoPosition) -> Player:
        return Player.FIRST if position.player.name == 'BLACK' else Player.SECOND

    def natural_terminal_wdl(self, position: NativeGoPosition) -> WdlTarget | None:
        if not position.is_terminal:
            return None
        if position.termination_reason.name == 'MAXIMUM_MOVES':
            return None
        value = position.terminal_value()
        if value > 0.0:
            return WdlTarget(win=1.0, draw=0.0, loss=0.0)
        if value < 0.0:
            return WdlTarget(win=0.0, draw=0.0, loss=1.0)
        return WdlTarget(win=0.0, draw=1.0, loss=0.0)

    def adjudicated_wdl(self, position: NativeGoPosition, reason: TerminationReason) -> WdlTarget:
        if reason not in {TerminationReason.MAXIMUM_PLIES, TerminationReason.ADJUDICATION}:
            raise ValueError(f'Go cannot adjudicate termination reason {reason.value}.')
        winner = position.area_score().winner
        if winner is None:
            return WdlTarget(win=0.0, draw=1.0, loss=0.0)
        if winner.name == position.player.name:
            return WdlTarget(win=1.0, draw=0.0, loss=0.0)
        return WdlTarget(win=0.0, draw=0.0, loss=1.0)

    def encode_network_input(self, position: NativeGoPosition) -> PackedPlanePayload:
        return self.packed_position(position)

    @property
    def augmentation_count(self) -> int:
        return 8

    def packed_position(self, position: NativeGoPosition) -> PackedPlanePayload:
        if position.board_size != self.board_size:
            raise ValueError('Native Go position board size disagrees with its Python contract.')
        return self.packed_planes.value(bytes(position.packed_encoding()))

    def transform_action_id(self, action_id: int, augmentation_index: int) -> int:
        try:
            symmetry = GoSymmetryIndex(augmentation_index)
        except ValueError as error:
            raise ValueError('Go augmentation index is outside the fixed layout.') from error
        if not 0 <= action_id < self.action_size:
            raise ValueError('Go action lies outside the action space.')
        if action_id == self.pass_action:
            return action_id
        x = action_id % self.board_size
        y = action_id // self.board_size
        maximum = self.board_size - 1
        match symmetry:
            case GoSymmetryIndex.IDENTITY:
                transformed_x, transformed_y = x, y
            case GoSymmetryIndex.ROTATE_90:
                transformed_x, transformed_y = maximum - y, x
            case GoSymmetryIndex.ROTATE_180:
                transformed_x, transformed_y = maximum - x, maximum - y
            case GoSymmetryIndex.ROTATE_270:
                transformed_x, transformed_y = y, maximum - x
            case GoSymmetryIndex.REFLECT:
                transformed_x, transformed_y = maximum - x, y
            case GoSymmetryIndex.REFLECT_ROTATE_90:
                transformed_x, transformed_y = maximum - y, maximum - x
            case GoSymmetryIndex.REFLECT_ROTATE_180:
                transformed_x, transformed_y = x, maximum - y
            case GoSymmetryIndex.REFLECT_ROTATE_270:
                transformed_x, transformed_y = y, x
        return transformed_y * self.board_size + transformed_x

    def transform_replay_targets(self, sample: ReplaySample, augmentation_index: int) -> ReplaySample:
        def transform_policy(policy: SparsePolicyTarget) -> SparsePolicyTarget:
            return SparsePolicyTarget(
                visits=tuple(
                    GameSearchVisit(
                        action_id=self.transform_action_id(visit.action_id, augmentation_index),
                        visit_count=visit.visit_count,
                    )
                    for visit in policy.visits
                ),
                legal_action_ids=tuple(
                    self.transform_action_id(action_id, augmentation_index) for action_id in policy.legal_action_ids
                ),
            )

        transformed_auxiliary = []
        for target in sample.auxiliary_targets:
            match target:
                case EligibleNextPolicyTarget(policy=policy):
                    transformed_auxiliary.append(EligibleNextPolicyTarget(policy=transform_policy(policy)))
                case EligibleRemainingGameLengthTarget() | EligibleScalarAuxiliaryTarget() | EligibleLegalMovesTarget():
                    transformed_auxiliary.append(target)
                case _:
                    transformed_auxiliary.append(target)
        return ReplaySample(
            encoded_state=self.transform_encoded_state(sample.encoded_state, augmentation_index),
            policy=transform_policy(sample.policy),
            wdl_target=sample.wdl_target,
            root_value=sample.root_value,
            auxiliary_targets=tuple(transformed_auxiliary),
            sample_weight=sample.sample_weight,
            source_model_generation=sample.source_model_generation,
            source_created_at_seconds=sample.source_created_at_seconds,
        )

    def transform_encoded_state(
        self,
        encoded_state: PackedPlanePayload,
        augmentation_index: int,
    ) -> PackedPlanePayload:
        try:
            symmetry = GoSymmetryIndex(augmentation_index)
        except ValueError as error:
            raise ValueError('Go augmentation index is outside the fixed layout.') from error
        state = decode_packed_planes(
            encoded_state,
            self.packed_planes,
            self.binary_channels,
            self.scalar_channels,
        )
        binary = state[: len(self.binary_channels)]
        rotation = int(symmetry) % 4
        if int(symmetry) >= int(GoSymmetryIndex.REFLECT):
            binary = np.flip(binary, axis=2)
        if rotation:
            binary = np.rot90(binary, k=-rotation, axes=(1, 2))
        transformed = state.copy()
        transformed[: len(self.binary_channels)] = binary
        return encode_packed_planes(
            transformed,
            self.packed_planes,
            self.binary_channels,
            self.scalar_channels,
        )

    def decode_batch_into(
        self,
        states: tuple[PackedPlanePayload, ...],
        output: npt.NDArray[np.float32],
    ) -> None:
        decode_packed_planes_into(
            states,
            self.packed_planes,
            self.binary_channels,
            self.scalar_channels,
            output,
        )
