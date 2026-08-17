from argparse import Namespace
from pathlib import Path

import pytest
import numpy as np

from src.training.architecture_benchmark import load_architecture_benchmark_plan
from tools.benchmark_chess_architectures import _run_benchmark
from tools.create_synthetic_architecture_replay import ReplayGenerationArguments, create_synthetic_architecture_replay


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


def test_contended_benchmark_plan_uses_two_minute_measurement() -> None:
    plan = load_architecture_benchmark_plan(Path('configs/benchmarks/chess-architecture-contended-120s.yaml'))

    assert plan.topology.trainer_device_ids == tuple(range(8))
    assert plan.topology.global_training_batch_size == 2048
    assert plan.topology.production_inference_batch_size == 64
    assert plan.training.equal_wall_time_seconds == 120.0


def test_chess_architecture_benchmark_refuses_gpu_work_without_acknowledgement() -> None:
    plan = load_architecture_benchmark_plan(Path('configs/benchmarks/chess-architecture-v1.yaml'))
    arguments = Namespace(acknowledge_gpu_load=False)

    with pytest.raises(ValueError, match='acknowledge-gpu-load'):
        _run_benchmark(arguments, plan)


def test_synthetic_architecture_replay_is_deterministic(tmp_path: Path) -> None:
    first_path = tmp_path / 'first.npz'
    second_path = tmp_path / 'second.npz'
    for output_path in (first_path, second_path):
        create_synthetic_architecture_replay(
            ReplayGenerationArguments(
                catalog_path=Path('configs/architectures/chess-cnn-attention-v1.yaml'),
                output_path=output_path,
                sample_count=4,
                random_seed=17,
            )
        )

    assert first_path.read_bytes() == second_path.read_bytes()
    with np.load(first_path, allow_pickle=False) as replay:
        assert replay['states'].shape == (4, 29, 8, 8)
        assert replay['policy_targets'].shape == (4, 1880)
        assert replay['wdl_targets'].shape == (4, 3)
        assert replay['next_policy_targets'].shape == (4, 1880)
        assert replay['remaining_length_targets'].shape == (4, 1)
