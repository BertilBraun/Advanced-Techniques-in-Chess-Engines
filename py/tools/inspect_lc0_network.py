"""Reports an Lc0 ONNX export's shapes and refuses the ones this branch cannot encode.

The encoder implements Lc0's classical 112-plane input. Networks declaring a canonicalized or
castling-plane input format need a different encoder, and a wrong guess produces a teacher that
looks merely weak rather than broken, so the mismatch is caught here instead of in match results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

EXPECTED_INPUT_PLANES = 112
EXPECTED_POLICY_SIZE = 1858
EXPECTED_WDL_SIZE = 3


def describe(onnx_path: Path) -> None:
    try:
        import onnx
    except ImportError as error:
        raise SystemExit('onnx is required to inspect an Lc0 export.') from error

    model = onnx.load(str(onnx_path))

    def shape_of(value) -> list[int | str]:
        return [
            dimension.dim_value if dimension.HasField('dim_value') else dimension.dim_param
            for dimension in value.type.tensor_type.shape.dim
        ]

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializer_names]
    print(f'Producer: {model.producer_name} {model.producer_version}')
    for value in inputs:
        print(f'  input  {value.name}: {shape_of(value)}')
    for value in model.graph.output:
        print(f'  output {value.name}: {shape_of(value)}')

    parameter_count = sum(_tensor_size(initializer) for initializer in model.graph.initializer)
    print(f'Parameter count: {parameter_count:,}')

    if len(inputs) != 1:
        raise SystemExit(f'Expected exactly one network input, found {len(inputs)}.')
    input_shape = shape_of(inputs[0])
    if len(input_shape) != 4 or input_shape[1] != EXPECTED_INPUT_PLANES:
        raise SystemExit(
            f'Input shape {input_shape} is not (batch, {EXPECTED_INPUT_PLANES}, 8, 8). This network does not use '
            'the classical 112-plane input, so the encoder on this branch cannot feed it.'
        )

    output_shapes = [shape_of(value) for value in model.graph.output]
    policy_outputs = [shape for shape in output_shapes if len(shape) == 2 and shape[1] == EXPECTED_POLICY_SIZE]
    wdl_outputs = [shape for shape in output_shapes if len(shape) == 2 and shape[1] == EXPECTED_WDL_SIZE]
    if not policy_outputs:
        raise SystemExit(f'No {EXPECTED_POLICY_SIZE}-wide policy output found among {output_shapes}.')
    if not wdl_outputs:
        raise SystemExit(
            f'No {EXPECTED_WDL_SIZE}-wide WDL output found among {output_shapes}. A value-only network cannot '
            'supply the WDL triple the search requires.'
        )
    print('Network is compatible with the classical 112-plane encoder on this branch.')


def _tensor_size(initializer) -> int:
    size = 1
    for dimension in initializer.dims:
        size *= dimension
    return size


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', type=Path, required=True, help='Network exported with `lc0 leela2onnx`.')
    return parser.parse_args()


def main() -> None:
    describe(parse_arguments().onnx)


if __name__ == '__main__':
    main()
