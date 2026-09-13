from __future__ import annotations

import argparse
from pathlib import Path

from AlphaZeroCpp import (
    AnalysisParameters,
    BatchedInferenceParameters,
    ChessAnalysis,
    InferenceBackend,
    InferenceConfiguration,
    InferenceDevice,
)


def preflight(engine_path: Path, batch_size: int, device_id: int) -> None:
    if not engine_path.is_file():
        raise ValueError(f'TensorRT template does not exist: {engine_path}')
    ChessAnalysis(
        InferenceConfiguration(
            device_id=device_id,
            model_path=str(engine_path),
            device=InferenceDevice.CUDA,
            backend=InferenceBackend.TENSORRT,
        ),
        AnalysisParameters(1, 1.5, BatchedInferenceParameters(1, batch_size, 2)),
    )
    print(f'Native TensorRT preflight passed with {engine_path}.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Verify native TensorRT inference construction before a run.')
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--device-id', type=int, default=0)
    arguments = parser.parse_args()
    preflight(arguments.engine, arguments.batch_size, arguments.device_id)


if __name__ == '__main__':
    main()
