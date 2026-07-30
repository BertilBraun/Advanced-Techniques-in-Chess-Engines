from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from src.az.games.go.configuration import NATIVE_INT32_MAX
from src.az.games.go.samples import DensePolicyTarget, GoSample, PolicyTarget, SparsePolicyTarget


class GoSymmetry(IntEnum):
    IDENTITY = 0
    ROTATE_90 = 1
    ROTATE_180 = 2
    ROTATE_270 = 3
    REFLECT = 4
    REFLECT_ROTATE_90 = 5
    REFLECT_ROTATE_180 = 6
    REFLECT_ROTATE_270 = 7


@dataclass(frozen=True)
class GoCoordinate:
    row: int
    column: int


def _validate_board_size(board_size: int) -> None:
    if board_size < 3:
        raise ValueError('Go symmetry board size must be at least 3.')
    if board_size * board_size >= NATIVE_INT32_MAX:
        raise ValueError('The Go board area and pass action must fit in a signed 32-bit integer.')


def inverse_symmetry(symmetry: GoSymmetry) -> GoSymmetry:
    match symmetry:
        case GoSymmetry.ROTATE_90:
            return GoSymmetry.ROTATE_270
        case GoSymmetry.ROTATE_270:
            return GoSymmetry.ROTATE_90
        case _:
            return symmetry


def transform_coordinate(row: int, column: int, board_size: int, symmetry: GoSymmetry) -> GoCoordinate:
    _validate_board_size(board_size)
    if not 0 <= row < board_size or not 0 <= column < board_size:
        raise ValueError('Go symmetry coordinate is outside the board.')
    maximum = board_size - 1
    match symmetry:
        case GoSymmetry.IDENTITY:
            return GoCoordinate(row, column)
        case GoSymmetry.ROTATE_90:
            return GoCoordinate(column, maximum - row)
        case GoSymmetry.ROTATE_180:
            return GoCoordinate(maximum - row, maximum - column)
        case GoSymmetry.ROTATE_270:
            return GoCoordinate(maximum - column, row)
        case GoSymmetry.REFLECT:
            return GoCoordinate(row, maximum - column)
        case GoSymmetry.REFLECT_ROTATE_90:
            return GoCoordinate(maximum - column, maximum - row)
        case GoSymmetry.REFLECT_ROTATE_180:
            return GoCoordinate(maximum - row, column)
        case GoSymmetry.REFLECT_ROTATE_270:
            return GoCoordinate(column, row)


def transform_action(action: int, board_size: int, symmetry: GoSymmetry) -> int:
    _validate_board_size(board_size)
    pass_action = board_size**2
    if action == pass_action:
        return action
    if not 0 <= action < pass_action:
        raise ValueError('Go action is outside the action space.')
    coordinate = transform_coordinate(action // board_size, action % board_size, board_size, symmetry)
    return coordinate.row * board_size + coordinate.column


def transform_planes(planes: np.ndarray, symmetry: GoSymmetry) -> np.ndarray:
    if planes.ndim != 3 or planes.shape[1] != planes.shape[2]:
        raise ValueError('Go planes must have shape planes x N x N.')
    transformed = np.empty_like(planes)
    board_size = planes.shape[1]
    _validate_board_size(board_size)
    for row in range(board_size):
        for column in range(board_size):
            target = transform_coordinate(row, column, board_size, symmetry)
            transformed[:, target.row, target.column] = planes[:, row, column]
    return transformed


def _transform_action_vector(values: np.ndarray, board_size: int, symmetry: GoSymmetry) -> np.ndarray:
    if values.ndim != 1 or len(values) != board_size**2 + 1:
        raise ValueError('Go action vector must have length N squared plus one.')
    transformed = np.empty_like(values)
    for action in range(len(values)):
        transformed[transform_action(action, board_size, symmetry)] = values[action]
    return transformed


def _transform_policy(policy: PolicyTarget, board_size: int, symmetry: GoSymmetry) -> PolicyTarget:
    match policy:
        case DensePolicyTarget(probabilities=probabilities):
            return DensePolicyTarget(_transform_action_vector(probabilities, board_size, symmetry))
        case SparsePolicyTarget(actions=actions, weights=weights):
            transformed_actions = np.asarray(
                [transform_action(int(action), board_size, symmetry) for action in actions],
                dtype=np.int32,
            )
            order = np.argsort(transformed_actions)
            return SparsePolicyTarget(transformed_actions[order], weights[order])


def transform_sample(sample: GoSample, symmetry: GoSymmetry) -> GoSample:
    board_size = sample.input_planes.shape[1]
    return GoSample(
        input_planes=transform_planes(sample.input_planes, symmetry),
        legal_action_mask=_transform_action_vector(sample.legal_action_mask, board_size, symmetry),
        policy_target=_transform_policy(sample.policy_target, board_size, symmetry),
        policy_weight=sample.policy_weight,
        value_target=sample.value_target,
        value_weight=sample.value_weight,
    )
