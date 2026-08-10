from pathlib import Path

from run_approved_experiment import approval_path
from src.experiment_queue.configuration import load_queue_configuration


def test_approval_path_is_selected_from_authored_configuration_name(tmp_path: Path) -> None:
    approval_directory = tmp_path / 'approvals'

    assert approval_path(approval_directory, Path('configs/10-random-opening-12.yaml')) == (
        approval_directory / '10-random-opening-12.json'
    )


def test_checked_in_screening_queue_owns_four_resource_slots_and_pending_order() -> None:
    configuration = load_queue_configuration(Path('configs/queues/vast-go-7x7-screening.yaml'))

    assert tuple(slot.cuda_devices for slot in configuration.slots) == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert tuple(slot.cpu_affinity for slot in configuration.slots) == (
        (24, 25, 26, 27, 28, 29, 30, 31, 56, 57, 58, 59, 60, 61, 62, 63),
        (16, 17, 18, 19, 20, 21, 22, 23, 48, 49, 50, 51, 52, 53, 54, 55),
        (8, 9, 10, 11, 12, 13, 14, 15, 40, 41, 42, 43, 44, 45, 46, 47),
        (0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39),
    )
    assert tuple(experiment.experiment_id for experiment in configuration.experiments) == (
        'go7-r13-00-baseline',
        'go7-r13-01-learning-rate-decay',
        'go7-r13-02-learning-rate-constant-004',
        'go7-r13-04-mixed-search-25-full',
        'go7-r13-06-progressive-search-64-512',
        'go7-r13-07-root-value-blend',
        'go7-r13-08-replay-ratio-8',
        'go7-r13-09-tree-retention-60',
        'go7-r13-10-random-opening-12',
        'go7-r13-11-next-policy',
    )
    assert configuration.wait_for_updates_when_empty
