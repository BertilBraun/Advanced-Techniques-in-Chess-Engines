from __future__ import annotations

from collections.abc import Callable

import pytest
from tools.tensorrt_quantization_recovery import identify_modelopt_autotune_recovery


def _raise_from_remove_partial_input_qdq(message: str) -> None:
    def remove_partial_input_qdq() -> None:
        raise IndexError(message)

    remove_partial_input_qdq()


def _raise_from_unrelated_function(message: str) -> None:
    raise IndexError(message)


def test_identifies_known_modelopt_postprocessing_failure() -> None:
    with pytest.raises(IndexError) as captured:
        _raise_from_remove_partial_input_qdq('list index out of range')

    recovery = identify_modelopt_autotune_recovery(captured.value)

    assert recovery is not None
    assert recovery.condition == 'remove_partial_input_qdq_index_error'
    assert recovery.message == 'list index out of range'


@pytest.mark.parametrize(
    ('raising_function', 'message'),
    (
        (_raise_from_remove_partial_input_qdq, 'different index error'),
        (_raise_from_unrelated_function, 'list index out of range'),
    ),
)
def test_rejects_other_index_errors(raising_function: Callable[[str], None], message: str) -> None:
    with pytest.raises(IndexError) as captured:
        raising_function(message)

    assert identify_modelopt_autotune_recovery(captured.value) is None


def test_rejects_other_exception_types() -> None:
    error = ValueError('list index out of range')

    assert identify_modelopt_autotune_recovery(error) is None
