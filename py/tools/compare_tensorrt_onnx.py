from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import Field
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from tools.benchmark_tensorrt_inference import _measure_runner, _TensorRtCudaGraphRunner
from tools.distill_train_student import (
    OpenedProductionReplay,
    ProductionReplayInput,
    close_training_dataset,
    dataset_split,
    open_training_dataset,
)
from tools.run_int8_architecture_screen import (
    _build_strongly_typed_engine,
    _held_out_batches,
    _legal_action_mask,
    _onnx_outputs,
    _tensorrt_outputs,
)
from tools.tensorrt_benchmark_metrics import FidelityMetrics, TimingDistribution, measure_fidelity

BATCH_SIZE = 320


@dataclass(frozen=True)
class Arguments:
    replay_store: Path
    replay_experiment: Path
    replay_sha256: str
    onnx: Path
    output: Path
    device_id: int
    positions: int
    holdout_fraction: float


class StronglyTypedComparison(FrozenModel):
    schema_version: int = 1
    onnx_path: str = Field(min_length=1)
    onnx_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    engine_path: str = Field(min_length=1)
    engine_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    replay_store_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    positions: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    engine_build_seconds: float = Field(ge=0.0)
    fidelity_to_onnx_runtime: FidelityMetrics
    timing: TimingDistribution


def run(arguments: Arguments) -> StronglyTypedComparison:
    dataset_input = ProductionReplayInput(
        kind='production_replay',
        path=arguments.replay_store,
        experiment=arguments.replay_experiment,
        orchestrator_recorded_sha256=arguments.replay_sha256,
    )
    opened = open_training_dataset(dataset_input)
    if not isinstance(opened, OpenedProductionReplay):
        raise AssertionError('TensorRT comparison requires a production replay store.')
    try:
        split = dataset_split(opened.row_count, arguments.holdout_fraction, 1.0)
        if arguments.positions > split.held_out_row_count:
            raise ValueError('Requested positions exceed the untouched holdout.')
        device = torch.device('cuda', arguments.device_id)
        batches = _held_out_batches(
            opened,
            split.held_out_start_row,
            arguments.positions,
            BATCH_SIZE,
            device,
        )
        legal_mask = torch.cat(tuple(_legal_action_mask(batch.policy_legal_action_ids) for batch in batches))
        reference = _onnx_outputs(arguments.onnx, batches, arguments.device_id)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        build_started = time.perf_counter()
        engine = _build_strongly_typed_engine(arguments.onnx, arguments.output)
        engine_build_seconds = time.perf_counter() - build_started
        timing_states = batches[0].states
        runner = _TensorRtCudaGraphRunner(arguments.output, timing_states, device, 10)
        candidate = _tensorrt_outputs(runner, batches)
        timing = _measure_runner(runner, 10, 5, 100, device)
        report = StronglyTypedComparison(
            onnx_path=str(arguments.onnx),
            onnx_sha256=file_sha256(arguments.onnx),
            engine_path=engine.path,
            engine_sha256=engine.sha256,
            replay_store_sha256=arguments.replay_sha256,
            positions=arguments.positions,
            batch_size=BATCH_SIZE,
            engine_build_seconds=engine_build_seconds,
            fidelity_to_onnx_runtime=measure_fidelity(reference, candidate, legal_mask),
            timing=timing,
        )
        report_path = arguments.output.with_suffix('.report.json')
        write_text_atomically(report_path, report.model_dump_json(indent=2) + '\n')
        return report
    finally:
        close_training_dataset(opened)


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Compare a strongly typed TensorRT engine with ONNX Runtime.')
    parser.add_argument('--replay-store', required=True, type=Path)
    parser.add_argument('--replay-experiment', required=True, type=Path)
    parser.add_argument('--replay-sha256', required=True)
    parser.add_argument('--onnx', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device-id', required=True, type=int)
    parser.add_argument('--positions', default=51_200, type=int)
    parser.add_argument('--holdout-fraction', default=0.02, type=float)
    namespace = parser.parse_args()
    if namespace.positions <= 0 or namespace.positions % BATCH_SIZE:
        raise ValueError(f'Positions must be a positive multiple of {BATCH_SIZE}.')
    return Arguments(
        replay_store=namespace.replay_store,
        replay_experiment=namespace.replay_experiment,
        replay_sha256=namespace.replay_sha256,
        onnx=namespace.onnx,
        output=namespace.output,
        device_id=namespace.device_id,
        positions=namespace.positions,
        holdout_fraction=namespace.holdout_fraction,
    )


def main() -> None:
    print(run(parse_arguments()).model_dump_json(indent=2))


if __name__ == '__main__':
    main()
