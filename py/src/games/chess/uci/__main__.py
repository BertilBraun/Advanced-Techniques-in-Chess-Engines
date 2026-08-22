from __future__ import annotations

import argparse

from src.games.chess.interactive.configuration import (
    DEFAULT_EXPLORATION_CONSTANT,
    InferenceTarget,
    InteractiveEngineConfiguration,
)
from src.games.chess.interactive.engine import InteractiveEngine
from src.games.chess.uci.engine import UciConfiguration, UciEngine
from src.games.chess.uci.server import UciServer
from src.self_play.configuration import SdpaBackend


def parse_arguments() -> tuple[InteractiveEngineConfiguration, UciConfiguration]:
    parser = argparse.ArgumentParser(description='Run AlphaZeroCpp as a UCI engine.')
    parser.add_argument('--model', required=True, help='TorchScript model path.')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--parallel-searches', type=int, default=64)
    parser.add_argument('--c-param', type=float, default=DEFAULT_EXPLORATION_CONSTANT)
    parser.add_argument('--inference-workers', type=int, default=2)
    parser.add_argument('--outstanding-batches-per-worker', type=int, default=2)
    parser.add_argument('--maximum-batch-size', type=int, default=64)
    parser.add_argument(
        '--sdpa-backend',
        choices=tuple(backend.value for backend in SdpaBackend),
        default=SdpaBackend.AUTOMATIC.value,
    )
    parser.add_argument('--search-slice-seconds', type=int, choices=range(1, 31), default=5)
    parser.add_argument(
        '--inference-target',
        choices=tuple(target.value for target in InferenceTarget),
        default=InferenceTarget.CUDA.value,
    )
    arguments = parser.parse_args()
    engine_configuration = InteractiveEngineConfiguration(
        model_path=arguments.model,
        device_id=arguments.device_id,
        parallel_searches=arguments.parallel_searches,
        exploration_constant=arguments.c_param,
        inference_workers=arguments.inference_workers,
        outstanding_batches_per_worker=arguments.outstanding_batches_per_worker,
        maximum_batch_size=arguments.maximum_batch_size,
        inference_target=InferenceTarget(arguments.inference_target),
        sdpa_backend=SdpaBackend(arguments.sdpa_backend),
    )
    return engine_configuration, UciConfiguration(search_slice_seconds=arguments.search_slice_seconds)


def main() -> None:
    engine_configuration, uci_configuration = parse_arguments()
    engine = InteractiveEngine(engine_configuration)
    UciServer(UciEngine(engine, uci_configuration)).run()


if __name__ == '__main__':
    main()
