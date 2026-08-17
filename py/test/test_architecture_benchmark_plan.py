from argparse import Namespace
from pathlib import Path

import pytest

from src.training.architecture_benchmark import load_architecture_benchmark_plan
from tools.benchmark_chess_architectures import _run_benchmark


def test_chess_architecture_benchmark_plan_matches_production_topology() -> None:
    plan = load_architecture_benchmark_plan(Path('configs/benchmarks/chess-architecture-v1.yaml'))

    assert plan.topology.trainer_device_ids == tuple(range(8))
    assert plan.topology.global_training_batch_size == 2048
    assert plan.topology.local_training_batch_size == 256
    assert plan.topology.self_play_processes_per_device == 2
    assert plan.topology.inference_workers_per_process == 2
    assert plan.topology.outstanding_batches_per_worker == 2
    assert 64 in plan.inference.batch_sizes
    assert plan.training.equal_sample_optimizer_steps * plan.topology.global_training_batch_size == 262144
    assert plan.training.equal_wall_time_seconds == 1800.0


def test_chess_architecture_benchmark_refuses_gpu_work_without_acknowledgement() -> None:
    plan = load_architecture_benchmark_plan(Path('configs/benchmarks/chess-architecture-v1.yaml'))
    arguments = Namespace(acknowledge_gpu_load=False)

    with pytest.raises(ValueError, match='acknowledge-gpu-load'):
        _run_benchmark(arguments, plan)
