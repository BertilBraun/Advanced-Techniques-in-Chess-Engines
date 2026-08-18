from argparse import Namespace
from pathlib import Path

import pytest
import numpy as np
import torch

from src.training.architecture_benchmark import load_architecture_benchmark_plan
from src.training.architecture_catalog import load_architecture_catalog
from src.training.network import Network
from tools.benchmark_chess_architectures import _create_inference_states, _prepare_inference_network, _run_benchmark
from tools.prepare_architecture_benchmark_model import (
    ModelPreparationArguments,
    prepare_architecture_benchmark_model,
)
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


def test_uncontended_smoke_plan_is_short_and_uses_production_inference_batch() -> None:
    plan = load_architecture_benchmark_plan(Path('configs/benchmarks/chess-architecture-uncontended-15s.yaml'))

    assert plan.topology.global_training_batch_size == 2048
    assert plan.training.equal_wall_time_seconds == 15.0
    assert plan.inference.batch_sizes == (64,)


def test_single_gpu_benchmark_plan_preserves_global_training_and_inference_batches() -> None:
    plan = load_architecture_benchmark_plan(Path('configs/benchmarks/chess-architecture-single-gpu-15s.yaml'))

    assert plan.topology.trainer_device_ids == (0,)
    assert plan.topology.global_training_batch_size == 2048
    assert plan.topology.local_training_batch_size == 2048
    assert plan.topology.production_inference_batch_size == 64
    assert plan.training.equal_wall_time_seconds == 15.0


def test_chess_architecture_benchmark_refuses_gpu_work_without_acknowledgement() -> None:
    plan = load_architecture_benchmark_plan(Path('configs/benchmarks/chess-architecture-v1.yaml'))
    arguments = Namespace(acknowledge_gpu_load=False)

    with pytest.raises(ValueError, match='acknowledge-gpu-load'):
        _run_benchmark(arguments, plan)


def test_inference_measurement_uses_bfloat16_model_and_inputs() -> None:
    plan = load_architecture_benchmark_plan(Path('configs/benchmarks/chess-architecture-uncontended-15s.yaml'))
    catalog = load_architecture_catalog(plan.catalog_path)
    definition = catalog.models[0].definition
    network = Network(
        definition.architecture,
        torch.device('cpu'),
        definition.dimensions,
        definition.auxiliary_output_sizes,
    )
    inference_network = _prepare_inference_network(network)
    states = _create_inference_states(network, batch_size=64, device=torch.device('cpu'))

    assert states.dtype is torch.bfloat16
    assert states.shape == (64, 29, 8, 8)
    assert {parameter.dtype for parameter in inference_network.parameters()} == {torch.bfloat16}


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


def test_architecture_benchmark_model_export_has_primary_inference_heads(tmp_path: Path) -> None:
    output_path = tmp_path / 'attention.jit.pt'
    parameter_count = prepare_architecture_benchmark_model(
        ModelPreparationArguments(
            catalog_path=Path('configs/architectures/chess-cnn-attention-v1.yaml'),
            model_id='chess-attention-1m',
            output_path=output_path,
            random_seed=7,
        )
    )
    model = torch.jit.load(str(output_path), map_location='cpu')
    policy, wdl = model(torch.zeros(2, 29, 8, 8))

    assert parameter_count == 1043856
    assert policy.shape == (2, 1880)
    assert wdl.shape == (2, 3)
