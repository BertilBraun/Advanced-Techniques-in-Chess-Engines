from pathlib import Path

import pytest
from pydantic import ValidationError

from src.experiment_queue.configuration import (
    QueueConfiguration,
    QueuedExperiment,
    ResourceRequest,
    ResourceSlot,
    RunnerCommand,
)
from src.experiment_queue.scheduler import create_assignment, schedule_experiments


def _slot(
    slot_id: str,
    cuda_devices: tuple[int, ...],
    cpu_affinity: tuple[int, ...],
    ram_capacity_bytes: int = 16_000,
) -> ResourceSlot:
    return ResourceSlot(
        slot_id=slot_id,
        cuda_devices=cuda_devices,
        cpu_affinity=cpu_affinity,
        ram_capacity_bytes=ram_capacity_bytes,
        working_directory=Path('/work') / slot_id,
        log_directory=Path('/logs') / slot_id,
    )


def _experiment(
    experiment_id: str,
    cuda_device_count: int,
    cpu_core_count: int = 2,
    ram_limit_bytes: int = 8_000,
) -> QueuedExperiment:
    return QueuedExperiment(
        experiment_id=experiment_id,
        experiment_file=Path('/experiments') / f'{experiment_id}.yaml',
        resources=ResourceRequest(
            cuda_device_count=cuda_device_count,
            cpu_core_count=cpu_core_count,
            ram_limit_bytes=ram_limit_bytes,
        ),
    )


def test_scheduler_assigns_queue_order_to_the_first_compatible_free_slots() -> None:
    slots = (
        _slot('gpu-pair', (0, 1), (0, 1, 2, 3)),
        _slot('gpu-single', (2,), (4, 5)),
        _slot('cpu', (), (6, 7)),
    )
    experiments = (
        _experiment('single-first', 1),
        _experiment('pair-second', 2, cpu_core_count=3),
        _experiment('single-waits', 1),
        _experiment('cpu-fourth', 0),
    )

    scheduled = schedule_experiments(experiments, slots)

    assert tuple(item.experiment.experiment_id for item in scheduled) == ('single-first', 'pair-second', 'cpu-fourth')
    assert tuple(item.assignment.slot_id for item in scheduled) == ('gpu-single', 'gpu-pair', 'cpu')
    assert scheduled[1].assignment.cpu_affinity == (0, 1, 2)


def test_scheduler_is_deterministic_and_never_allocates_one_slot_twice() -> None:
    slots = (_slot('first', (0,), (0, 1)), _slot('second', (1,), (2, 3)))
    experiments = tuple(_experiment(f'experiment-{index}', 1) for index in range(4))

    first_schedule = schedule_experiments(experiments, slots)
    second_schedule = schedule_experiments(experiments, slots)

    assert first_schedule == second_schedule
    assert tuple(item.assignment.slot_id for item in first_schedule) == ('first', 'second')


def test_assignment_rejects_an_incompatible_slot() -> None:
    with pytest.raises(ValueError, match='cannot satisfy'):
        create_assignment(_experiment('needs-pair', 2), _slot('single', (0,), (0, 1)))


@pytest.mark.parametrize(
    ('field_name', 'value'),
    (
        ('cuda_devices', (0, 0)),
        ('cpu_affinity', (2, 1)),
        ('cpu_affinity', (-1, 0)),
    ),
)
def test_resource_slot_rejects_invalid_device_sets(field_name: str, value: tuple[int, ...]) -> None:
    candidate = _slot('slot', (0,), (0, 1)).model_dump()
    candidate[field_name] = value

    with pytest.raises(ValidationError, match='unique and strictly increasing|nonnegative'):
        ResourceSlot.model_validate(candidate)


def test_queue_rejects_overlapping_slots_and_unassignable_requests() -> None:
    runner = RunnerCommand(command=('python', 'py/train.py'))

    with pytest.raises(ValidationError, match='share CUDA devices'):
        QueueConfiguration(
            runner=runner,
            slots=(_slot('first', (0,), (0, 1)), _slot('second', (0,), (2, 3))),
            experiments=(_experiment('experiment', 1),),
            summary_path=Path('/summary.json'),
        )

    with pytest.raises(ValidationError, match='no compatible resource slot'):
        QueueConfiguration(
            runner=runner,
            slots=(_slot('single', (0,), (0, 1)),),
            experiments=(_experiment('pair', 2),),
            summary_path=Path('/summary.json'),
        )


def test_queue_models_are_frozen_and_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError, match='Extra inputs are not permitted'):
        ResourceRequest.model_validate(
            {
                'cuda_device_count': 1,
                'cpu_core_count': 2,
                'ram_limit_bytes': 8_000,
                'priority': 'urgent',
            }
        )
