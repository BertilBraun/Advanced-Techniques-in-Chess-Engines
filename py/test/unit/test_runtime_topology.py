from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.az.config.profiles import default_topology
from src.az.config.runtime import TopologyConfiguration
from src.az.runtime.topology import RuntimeTopology, WorkerAssignment


def test_runtime_topology_validates_contiguous_workers_and_visible_devices() -> None:
    topology = RuntimeTopology(
        workers=(
            WorkerAssignment(
                worker_index=0,
                device_worker_index=0,
                device_index=0,
                search_thread_count=4,
                maximum_active_searches=4,
            ),
            WorkerAssignment(
                worker_index=1,
                device_worker_index=0,
                device_index=1,
                search_thread_count=4,
                maximum_active_searches=4,
            ),
        ),
        optimizer_active_worker_ids=(0, 1),
        trainer_device_indices=(0, 1),
    )

    topology.validate_visible_cuda_devices(2)

    with pytest.raises(ValueError, match="not visible"):
        topology.validate_visible_cuda_devices(1)


def test_runtime_topology_rejects_noncontiguous_worker_indices() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        RuntimeTopology(
            workers=(
                WorkerAssignment(
                    worker_index=1,
                    device_worker_index=0,
                    device_index=None,
                    search_thread_count=1,
                    maximum_active_searches=1,
                ),
            ),
            optimizer_active_worker_ids=(0,),
            trainer_device_indices=(),
        )


def test_runtime_topology_exposes_optimizer_phase_worker_partition() -> None:
    topology = RuntimeTopology(
        workers=tuple(
            WorkerAssignment(
                worker_index=worker_index,
                device_worker_index=worker_index % 4,
                device_index=worker_index // 4,
                search_thread_count=4,
                maximum_active_searches=64,
            )
            for worker_index in range(8)
        ),
        optimizer_active_worker_ids=(0, 4),
        trainer_device_indices=(0, 1),
    )

    assert topology.optimizer_active_worker_ids == (0, 4)
    assert topology.optimizer_paused_worker_ids == (1, 2, 3, 5, 6, 7)


def test_resolved_topology_exposes_reference_process_and_thread_allocation() -> None:
    topology = default_topology()

    assert topology.self_play_workers_per_device == 4
    assert topology.search_threads_per_worker == 4
    assert topology.self_play_worker_count == 8
    assert topology.worker_ids_for_device_position(0) == (0, 1, 2, 3)
    assert topology.worker_ids_for_device_position(1) == (4, 5, 6, 7)
    assert topology.optimizer_active_self_play_worker_ids == (0, 4)
    assert topology.optimizer_paused_self_play_worker_ids == (1, 2, 3, 5, 6, 7)


@pytest.mark.parametrize(
    "active_worker_ids",
    (
        (0,),
        (0, 1, 4),
        (0, 8),
        (4, 0),
    ),
)
def test_resolved_topology_rejects_invalid_optimizer_worker_partition(
    active_worker_ids: tuple[int, ...],
) -> None:
    candidate = default_topology().model_dump(mode="python")
    candidate["optimizer_active_self_play_worker_ids"] = active_worker_ids

    with pytest.raises(ValidationError, match="Optimizer-active|Exactly one"):
        TopologyConfiguration.model_validate(candidate)
