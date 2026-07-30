from __future__ import annotations

import pytest

from src.az.runtime.topology import RuntimeTopology, WorkerAssignment


def test_runtime_topology_validates_contiguous_workers_and_visible_devices() -> None:
    topology = RuntimeTopology(
        workers=(
            WorkerAssignment(worker_index=0, device_index=0, maximum_active_searches=4),
            WorkerAssignment(worker_index=1, device_index=1, maximum_active_searches=4),
        ),
        trainer_device_indices=(0, 1),
    )

    topology.validate_visible_cuda_devices(2)

    with pytest.raises(ValueError, match='not visible'):
        topology.validate_visible_cuda_devices(1)


def test_runtime_topology_rejects_noncontiguous_worker_indices() -> None:
    with pytest.raises(ValueError, match='contiguous'):
        RuntimeTopology(
            workers=(WorkerAssignment(worker_index=1, device_index=None, maximum_active_searches=1),),
            trainer_device_indices=(),
        )
