from __future__ import annotations

import math
import struct


def entropy_calibration_tensor_scale(cache: bytes, tensor_name: str) -> float:
    prefix = f'{tensor_name}: '.encode('ascii')
    matching_lines = tuple(line for line in cache.splitlines() if line.startswith(prefix))
    if len(matching_lines) != 1:
        raise ValueError(f'Calibration cache does not contain exactly one scale for {tensor_name}.')
    encoded_scale = matching_lines[0][len(prefix) :]
    if len(encoded_scale) != 8:
        raise ValueError(f'Calibration cache scale for {tensor_name} is not a 32-bit hexadecimal value.')
    try:
        scale = struct.unpack('>f', bytes.fromhex(encoded_scale.decode('ascii')))[0]
    except (UnicodeDecodeError, ValueError, struct.error) as error:
        raise ValueError(f'Calibration cache scale for {tensor_name} is invalid.') from error
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f'Calibration cache scale for {tensor_name} must be positive and finite.')
    return scale


def validate_input_calibration_scale(scale: float, observed_maximum_absolute_input: float) -> None:
    if not math.isfinite(observed_maximum_absolute_input) or observed_maximum_absolute_input <= 0.0:
        raise ValueError('Observed calibration input magnitude must be positive and finite.')
    if scale > observed_maximum_absolute_input * 2.0:
        raise ValueError(
            f'Calibration input scale {scale:g} is incompatible with observed maximum magnitude '
            f'{observed_maximum_absolute_input:g}; check the calibration device-buffer dtype.'
        )
