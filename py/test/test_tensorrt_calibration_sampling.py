from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import tools.tensorrt_calibration_sampling as sampling
from tools.tensorrt_calibration_sampling import EncodedStates, select_disjoint_replay_calibration


def _state_loader(indices: npt.NDArray[np.int64]) -> tuple[EncodedStates, npt.NDArray[np.int8]]:
    encoded_states = indices.astype(np.uint8).reshape(-1, 1)
    decoded_states = indices.astype(np.int8).reshape(-1, 1, 1, 1)
    return encoded_states, decoded_states


def test_replay_calibration_is_deterministic_and_excludes_fidelity_representations() -> None:
    excluded_states = np.asarray((1, 4), dtype=np.int8).reshape(-1, 1, 1, 1)

    first = select_disjoint_replay_calibration(8, 5, 17, excluded_states, _state_loader)
    second = select_disjoint_replay_calibration(8, 5, 17, excluded_states, _state_loader)

    np.testing.assert_array_equal(first.logical_indices, second.logical_indices)
    np.testing.assert_array_equal(first.encoded_states, second.encoded_states)
    np.testing.assert_array_equal(first.decoded_states, second.decoded_states)
    assert len(first.logical_indices) == 5
    assert len(np.unique(first.logical_indices)) == 5
    assert not np.any(np.isin(first.decoded_states, excluded_states))
    assert first.excluded_overlap_count == 2


def test_replay_calibration_fails_when_too_few_disjoint_representations_exist() -> None:
    excluded_states = np.arange(3, dtype=np.int8).reshape(-1, 1, 1, 1)

    with pytest.raises(ValueError, match='only 0 rows'):
        select_disjoint_replay_calibration(3, 2, 17, excluded_states, _state_loader)


def test_replay_calibration_refills_after_excluding_initial_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sampling, 'REFILL_CHUNK_SIZE', 2)
    excluded_states = np.asarray((0, 1, 2), dtype=np.int8).reshape(-1, 1, 1, 1)

    sample = select_disjoint_replay_calibration(6, 3, 17, excluded_states, _state_loader)

    np.testing.assert_array_equal(np.sort(sample.logical_indices), np.asarray((3, 4, 5)))
    assert sample.excluded_overlap_count == 3
