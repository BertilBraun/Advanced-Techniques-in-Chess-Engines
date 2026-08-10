from pathlib import Path

from run_approved_experiment import approval_path
from src.experiment_queue.configuration import load_queue_configuration


def test_approval_path_is_selected_from_authored_configuration_name(tmp_path: Path) -> None:
    approval_directory = tmp_path / 'approvals'

    assert approval_path(approval_directory, Path('configs/10-random-opening-12.yaml')) == (
        approval_directory / '10-random-opening-12.json'
    )


def test_checked_in_screening_queue_owns_two_resource_slots_and_pending_order() -> None:
    configuration = load_queue_configuration(Path('configs/queues/vast-go-7x7-screening.yaml'))

    assert tuple(slot.cuda_devices for slot in configuration.slots) == ((0, 1), (2, 3))
    assert tuple(experiment.experiment_id for experiment in configuration.experiments) == (
        'go7-00-baseline',
        'go7-01-learning-rate-decay',
        'go7-02-learning-rate-constant-004',
        'go7-04-mixed-search-25-full',
        'go7-06-progressive-search-64-512',
        'go7-07-root-value-blend',
        'go7-08-replay-ratio-8',
        'go7-09-tree-retention-60',
        'go7-10-random-opening-12',
        'go7-11-next-policy',
    )
    assert configuration.wait_for_updates_when_empty
