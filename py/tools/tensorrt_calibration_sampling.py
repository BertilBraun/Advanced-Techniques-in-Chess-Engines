from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

REFILL_CHUNK_SIZE = 1_024

EncodedStates = npt.NDArray[np.uint8]
DecodedStates = npt.NDArray[np.int8]
ReplayStateLoader = Callable[[npt.NDArray[np.int64]], tuple[EncodedStates, DecodedStates]]


@dataclass(frozen=True)
class ReplayCalibrationSample:
    logical_indices: npt.NDArray[np.int64]
    encoded_states: EncodedStates
    decoded_states: DecodedStates
    excluded_overlap_count: int


def select_disjoint_replay_calibration(
    available_positions: int,
    position_count: int,
    random_seed: int,
    excluded_states: DecodedStates,
    load_states: ReplayStateLoader,
) -> ReplayCalibrationSample:
    if position_count <= 0 or position_count > available_positions:
        raise ValueError('Calibration position count must fit within the available replay positions.')
    if random_seed < 0:
        raise ValueError('Calibration random seed must be nonnegative.')
    if excluded_states.ndim != 4 or excluded_states.shape[0] == 0:
        raise ValueError('Calibration sampling requires nonempty rank-four excluded states.')

    candidate_order = np.random.default_rng(random_seed).permutation(available_positions).astype(np.int64)
    excluded_digests = {_row_digest(row) for row in excluded_states}
    selected_indices: list[npt.NDArray[np.int64]] = []
    selected_encoded_states: list[EncodedStates] = []
    selected_decoded_states: list[DecodedStates] = []
    selected_count = 0
    excluded_overlap_count = 0
    candidate_offset = 0

    while selected_count < position_count and candidate_offset < available_positions:
        needed = position_count - selected_count
        candidate_count = min(available_positions - candidate_offset, max(needed, REFILL_CHUNK_SIZE))
        candidate_indices = candidate_order[candidate_offset : candidate_offset + candidate_count]
        candidate_offset += candidate_count
        gather_order = np.argsort(candidate_indices)
        gathered_indices = candidate_indices[gather_order]
        gathered_encoded_states, gathered_decoded_states = load_states(gathered_indices)
        _validate_loaded_states(
            gathered_indices,
            gathered_encoded_states,
            gathered_decoded_states,
            excluded_states.shape[1:],
        )
        candidate_order_restore = np.argsort(gather_order)
        encoded_states = gathered_encoded_states[candidate_order_restore]
        decoded_states = gathered_decoded_states[candidate_order_restore]
        keep = np.asarray([_row_digest(row) not in excluded_digests for row in decoded_states], dtype=np.bool_)
        excluded_overlap_count += int(np.count_nonzero(~keep))
        retained_indices = candidate_indices[keep][:needed]
        retained_count = len(retained_indices)
        if retained_count == 0:
            continue
        selected_indices.append(retained_indices)
        selected_encoded_states.append(encoded_states[keep][:needed])
        selected_decoded_states.append(decoded_states[keep][:needed])
        selected_count += retained_count

    if selected_count != position_count:
        raise ValueError(
            f'Replay contains only {selected_count} rows outside the excluded fidelity representations; '
            f'{position_count} are required.'
        )
    return ReplayCalibrationSample(
        logical_indices=np.concatenate(selected_indices),
        encoded_states=np.concatenate(selected_encoded_states),
        decoded_states=np.concatenate(selected_decoded_states),
        excluded_overlap_count=excluded_overlap_count,
    )


def _row_digest(row: npt.NDArray[np.generic]) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()


def _validate_loaded_states(
    indices: npt.NDArray[np.int64],
    encoded_states: EncodedStates,
    decoded_states: DecodedStates,
    expected_state_shape: tuple[int, ...],
) -> None:
    if encoded_states.ndim != 2 or encoded_states.shape[0] != len(indices):
        raise ValueError('Loaded packed replay states are not aligned with the requested indices.')
    if decoded_states.shape != (len(indices), *expected_state_shape):
        raise ValueError('Loaded decoded replay states do not match the fidelity representation shape.')
