from __future__ import annotations

import numpy as np
import pytest

from src.az.games.go.augmentation import (
    GoSymmetry,
    inverse_symmetry,
    transform_action,
    transform_coordinate,
    transform_planes,
    transform_sample,
)
from src.az.games.go.samples import DensePolicyTarget
from test.unit.go_stage5_helpers import sample


@pytest.mark.parametrize('symmetry', list(GoSymmetry))
@pytest.mark.parametrize('board_size', [7, 9])
def test_all_symmetries_round_trip_state_policy_and_actions(
    symmetry: GoSymmetry,
    board_size: int,
) -> None:
    original = sample(board_size)

    transformed = transform_sample(original, symmetry)
    restored = transform_sample(transformed, inverse_symmetry(symmetry))

    assert np.array_equal(restored.input_planes, original.input_planes)
    assert np.array_equal(restored.legal_action_mask, original.legal_action_mask)
    assert isinstance(restored.policy_target, DensePolicyTarget)
    assert isinstance(original.policy_target, DensePolicyTarget)
    assert np.array_equal(restored.policy_target.probabilities, original.policy_target.probabilities)
    for action in range(board_size**2 + 1):
        transformed_action = transform_action(action, board_size, symmetry)
        assert transform_action(transformed_action, board_size, inverse_symmetry(symmetry)) == action


@pytest.mark.parametrize('symmetry', list(GoSymmetry))
def test_pass_action_is_invariant(symmetry: GoSymmetry) -> None:
    assert transform_action(49, 7, symmetry) == 49


@pytest.mark.parametrize('board_size', [0, 6, 8, 10])
def test_symmetry_rejects_unsupported_board_sizes(board_size: int) -> None:
    with pytest.raises(ValueError, match='7 or 9'):
        transform_action(0, board_size, GoSymmetry.IDENTITY)


@pytest.mark.parametrize(('row', 'column'), [(-1, 0), (0, -1), (7, 0), (0, 7)])
def test_symmetry_rejects_out_of_bounds_coordinates(row: int, column: int) -> None:
    with pytest.raises(ValueError, match='outside'):
        transform_coordinate(row, column, 7, GoSymmetry.IDENTITY)


def test_symmetry_rejects_malformed_plane_shapes() -> None:
    with pytest.raises(ValueError, match='planes x N x N'):
        transform_planes(np.zeros((2, 7, 8), dtype=np.float32), GoSymmetry.IDENTITY)
    with pytest.raises(ValueError, match='7 or 9'):
        transform_planes(np.zeros((2, 8, 8), dtype=np.float32), GoSymmetry.IDENTITY)
