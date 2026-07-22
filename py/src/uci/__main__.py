from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.eval.InteractiveEngine import InferenceTarget
from src.settings import PLAY_C_PARAM
from src.uci.optimized import OptimizedEngineConfiguration, OptimizedUciEngine
from src.uci.server import UciServer


@dataclass(frozen=True)
class CommandLineArguments:
    model_path: str
    device_id: int
    parallel_searches: int
    exploration_constant: float
    inference_workers: int
    outstanding_batches_per_worker: int
    maximum_batch_size: int
    search_slice_seconds: int
    inference_target: InferenceTarget


def parse_arguments() -> CommandLineArguments:
    parser = argparse.ArgumentParser(description='Run AlphaZeroCpp as a UCI engine.')
    parser.add_argument('--model', required=True, help='TorchScript model path.')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--parallel-searches', type=int, default=64)
    parser.add_argument('--c-param', type=float, default=PLAY_C_PARAM)
    parser.add_argument('--inference-workers', type=int, default=2)
    parser.add_argument('--outstanding-batches-per-worker', type=int, default=2)
    parser.add_argument('--maximum-batch-size', type=int, default=64)
    parser.add_argument('--search-slice-seconds', type=int, choices=range(1, 31), default=5)
    parser.add_argument(
        '--inference-target',
        choices=tuple(target.value for target in InferenceTarget),
        default=InferenceTarget.CUDA.value,
    )
    arguments = parser.parse_args()
    return CommandLineArguments(
        model_path=arguments.model,
        device_id=arguments.device_id,
        parallel_searches=arguments.parallel_searches,
        exploration_constant=arguments.c_param,
        inference_workers=arguments.inference_workers,
        outstanding_batches_per_worker=arguments.outstanding_batches_per_worker,
        maximum_batch_size=arguments.maximum_batch_size,
        search_slice_seconds=arguments.search_slice_seconds,
        inference_target=InferenceTarget(arguments.inference_target),
    )


def main() -> None:
    arguments = parse_arguments()
    configuration = OptimizedEngineConfiguration(
        model_path=arguments.model_path,
        device_id=arguments.device_id,
        parallel_searches=arguments.parallel_searches,
        exploration_constant=arguments.exploration_constant,
        inference_workers=arguments.inference_workers,
        outstanding_batches_per_worker=arguments.outstanding_batches_per_worker,
        maximum_batch_size=arguments.maximum_batch_size,
        search_slice_seconds=arguments.search_slice_seconds,
        inference_target=arguments.inference_target,
    )
    UciServer(OptimizedUciEngine(configuration)).run()


if __name__ == '__main__':
    main()
