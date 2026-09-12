from __future__ import annotations

import pytest
from tools.tensorrt_calibration_cache import (
    entropy_calibration_tensor_scale,
    validate_input_calibration_scale,
)


def test_entropy_calibration_tensor_scale_decodes_tensor_rt_hex_float() -> None:
    cache = b'TRT-101401-EntropyCalibration2\nstates: 3c010204\n'

    assert entropy_calibration_tensor_scale(cache, 'states') == pytest.approx(1.0 / 127.0)


def test_input_calibration_scale_rejects_half_words_misread_as_float32() -> None:
    cache = b'TRT-101401-EntropyCalibration2\nstates: 5221c314\n'

    with pytest.raises(ValueError, match='device-buffer dtype'):
        validate_input_calibration_scale(entropy_calibration_tensor_scale(cache, 'states'), 127.0)


def test_entropy_calibration_tensor_scale_requires_the_named_tensor() -> None:
    with pytest.raises(ValueError, match='exactly one scale'):
        entropy_calibration_tensor_scale(b'TRT-101401-EntropyCalibration2\n', 'states')
