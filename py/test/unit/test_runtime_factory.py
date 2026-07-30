from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from src.az.config.resolution import resolve_configuration
from src.az.config.root import ResolvedRunConfiguration
from src.az.config.search import (
    MixedSearchBudget,
    ParentValueFpu,
    RetainSubtree,
)
from src.az.config.runtime import TelemetryMetric
from src.az.config.serialization import load_authoring_configuration
from src.az.runtime.factory import RuntimeBuildEnvironment, build_runtime_plan


def _configuration() -> ResolvedRunConfiguration:
    return resolve_configuration(load_authoring_configuration(Path('configs/v2/go-7x7-fixed.authoring.json')))


def _environment(tmp_path: Path) -> RuntimeBuildEnvironment:
    configuration = _configuration()
    return RuntimeBuildEnvironment(
        run_id=UUID(int=811),
        resolved_configuration_sha256='e' * 64,
        output_directory=tmp_path.resolve(),
        checkpoint_directory=(tmp_path / 'checkpoints').resolve(),
        startup_timeout_seconds=120,
        shutdown_grace_seconds=30,
        visible_cuda_models=(
            configuration.hardware.expected_gpu_model,
            configuration.hardware.expected_gpu_model,
        ),
        logical_cpu_count=configuration.hardware.minimum_logical_cpu_count,
        ram_gib=configuration.hardware.minimum_ram_gib,
        free_disk_gib=configuration.hardware.minimum_free_disk_gib,
        allow_cpu_smoke=False,
    )


def test_runtime_factory_consumes_fixed_baseline_configuration(
    tmp_path: Path,
) -> None:
    configuration = _configuration()
    plan = build_runtime_plan(configuration, _environment(tmp_path))

    assert len(plan.worker_specifications) == len(configuration.topology.self_play.device_ids)
    first = plan.worker_specifications[0]
    assert first.game_configuration == configuration.game
    assert first.model_configuration == configuration.model.schedule.architecture
    assert first.search.simulation_cap == configuration.search.budget.simulations
    assert first.search.exploration_constant == configuration.search.algorithm.exploration_constant
    assert first.search.backup_discount == configuration.search.backup_discount
    assert first.search.no_visited_child_value == configuration.search.fpu.no_visited_child_value
    assert first.search.root_exploration == configuration.search.root_exploration
    assert first.search.temperature == configuration.search.temperature
    assert first.logical_worker_count == configuration.topology.self_play_workers_per_device
    assert first.maximum_active_searches_per_worker == configuration.topology.maximum_active_searches_per_worker
    assert first.maximum_batch_size == configuration.search.inference.maximum_batch_size
    assert first.maximum_wait_microseconds == configuration.search.inference.maximum_wait_microseconds
    assert first.maximum_pending_batches == configuration.topology.maximum_pending_inference_batches
    assert first.inference_cache_capacity == configuration.search.inference.cache_capacity
    assert first.value_target_weight == configuration.self_play.value_target_weight
    assert first.telemetry_write_every_seconds == configuration.telemetry.write_every_seconds
    assert first.resource_sample_every_seconds == configuration.telemetry.resource_sample_every_seconds
    assert plan.games_per_shard == configuration.self_play.games_per_shard
    assert plan.duration_seconds == configuration.experiment.duration_seconds
    assert plan.startup_timeout_seconds == 120
    assert plan.required_metrics == configuration.telemetry.required_metrics
    assert plan.search_trace_sample_probability == configuration.telemetry.search_trace_sample_probability
    assert plan.replay_directory == tmp_path.resolve() / 'replay'
    assert plan.topology.trainer_device_indices == configuration.topology.trainer.device_ids
    assert plan.topology.evaluation_device_indices == configuration.topology.evaluation.device_ids
    assert tuple(specification.device for specification in plan.worker_specifications) == (
        'cuda:0',
        'cuda:1',
    )
    assert plan.topology.workers[0].maximum_active_searches == (
        configuration.topology.self_play_workers_per_device * configuration.topology.maximum_active_searches_per_worker
    )


@pytest.mark.parametrize(
    ('search_update', 'message'),
    (
        (
            {
                'budget': MixedSearchBudget(
                    kind='mixed',
                    cheap_simulations=2,
                    full_simulations=8,
                    full_search_probability=0.25,
                    cheap_policy_target_weight=0,
                    full_policy_target_weight=1,
                )
            },
            'fixed search budgets',
        ),
        ({'fpu': ParentValueFpu(kind='parent_value')}, 'visited-child-mean FPU'),
        (
            {
                'tree_reuse': RetainSubtree(
                    kind='retain_subtree',
                    maximum_retained_nodes=100,
                )
            },
            'disabled tree reuse',
        ),
    ),
)
def test_runtime_factory_rejects_stage8_search_features(
    tmp_path: Path,
    search_update: dict[str, object],
    message: str,
) -> None:
    configuration = _configuration()
    unsupported = configuration.model_copy(update={'search': configuration.search.model_copy(update=search_update)})

    with pytest.raises(ValueError, match=message):
        build_runtime_plan(unsupported, _environment(tmp_path))


def test_runtime_factory_rejects_absent_configured_hardware(
    tmp_path: Path,
) -> None:
    unavailable = RuntimeBuildEnvironment(
        run_id=UUID(int=812),
        resolved_configuration_sha256='f' * 64,
        output_directory=tmp_path.resolve(),
        checkpoint_directory=(tmp_path / 'checkpoints').resolve(),
        startup_timeout_seconds=120,
        shutdown_grace_seconds=30,
        visible_cuda_models=(),
        logical_cpu_count=_configuration().hardware.minimum_logical_cpu_count,
        ram_gib=_configuration().hardware.minimum_ram_gib,
        free_disk_gib=_configuration().hardware.minimum_free_disk_gib,
        allow_cpu_smoke=False,
    )

    with pytest.raises(ValueError, match='device count'):
        build_runtime_plan(_configuration(), unavailable)


def test_runtime_factory_rejects_insufficient_host_resources(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    insufficient = RuntimeBuildEnvironment(
        run_id=environment.run_id,
        resolved_configuration_sha256=environment.resolved_configuration_sha256,
        output_directory=environment.output_directory,
        checkpoint_directory=environment.checkpoint_directory,
        startup_timeout_seconds=environment.startup_timeout_seconds,
        shutdown_grace_seconds=environment.shutdown_grace_seconds,
        visible_cuda_models=environment.visible_cuda_models,
        logical_cpu_count=1,
        ram_gib=environment.ram_gib,
        free_disk_gib=environment.free_disk_gib,
        allow_cpu_smoke=False,
    )

    with pytest.raises(ValueError, match='logical CPU'):
        build_runtime_plan(_configuration(), insufficient)


def test_runtime_factory_rejects_unimplemented_telemetry_promises(tmp_path: Path) -> None:
    configuration = _configuration()
    traces = configuration.model_copy(
        update={'telemetry': configuration.telemetry.model_copy(update={'search_trace_sample_probability': 0.1})}
    )
    unsupported_metric = configuration.model_copy(
        update={
            'telemetry': configuration.telemetry.model_copy(
                update={'required_metrics': (TelemetryMetric.GPU_UTILIZATION,)}
            )
        }
    )

    with pytest.raises(ValueError, match='reserved for Stage 9'):
        build_runtime_plan(traces, _environment(tmp_path))
    with pytest.raises(ValueError, match='not derivable'):
        build_runtime_plan(unsupported_metric, _environment(tmp_path))


def test_active_search_capacity_is_driven_by_topology_configuration(tmp_path: Path) -> None:
    configuration = _configuration()
    altered = configuration.model_copy(
        update={'topology': configuration.topology.model_copy(update={'maximum_active_searches_per_worker': 3})}
    )

    plan = build_runtime_plan(altered, _environment(tmp_path))

    assert plan.worker_specifications[0].maximum_active_searches_per_worker == 3
    assert plan.topology.workers[0].maximum_active_searches == 3 * altered.topology.self_play_workers_per_device
