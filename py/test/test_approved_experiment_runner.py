from pathlib import Path

from run_approved_experiment import approval_path
from src.experiment_queue.configuration import load_queue_configuration


def test_approval_path_is_selected_from_authored_configuration_name(tmp_path: Path) -> None:
    approval_directory = tmp_path / 'approvals'

    assert approval_path(approval_directory, Path('configs/10-random-opening-12.yaml')) == (
        approval_directory / '10-random-opening-12.json'
    )


def _assert_eight_gpu_topology(configuration_path: Path) -> tuple[str, ...]:
    configuration = load_queue_configuration(configuration_path)

    assert tuple(slot.cuda_devices for slot in configuration.slots) == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert tuple(slot.cpu_affinity for slot in configuration.slots) == (
        (24, 25, 26, 27, 28, 29, 30, 31, 56, 57, 58, 59, 60, 61, 62, 63),
        (16, 17, 18, 19, 20, 21, 22, 23, 48, 49, 50, 51, 52, 53, 54, 55),
        (8, 9, 10, 11, 12, 13, 14, 15, 40, 41, 42, 43, 44, 45, 46, 47),
        (0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39),
    )
    assert configuration.wait_for_updates_when_empty
    return tuple(experiment.experiment_id for experiment in configuration.experiments)


def test_checked_in_go7_screening_queue_records_the_terminal_r14_order() -> None:
    assert _assert_eight_gpu_topology(Path('configs/queues/vast-go-7x7-screening.yaml')) == (
        'go7-r14-12-fpu-reduction-02',
        'go7-r14-13-restart-states',
        'go7-r14-14-remaining-game-length',
        'go7-r14-15-forced-playouts',
        'go7-r14-14b-remaining-game-length',
        'go7-r14-12b-fpu-reduction-02',
        'go7-r14-15b-forced-playouts',
        'go7-r14-13b-restart-states',
    )


def test_checked_in_go9_screening_queue_records_the_authored_ablation_order() -> None:
    assert _assert_eight_gpu_topology(Path('configs/queues/vast-go-9x9-screening.yaml')) == tuple(
        f'go9-r15c-{index:02d}-{name}'
        for index, name in enumerate(
            (
                'baseline',
                'zero-fpu',
                'no-forced-playouts',
                'squeeze-excitation',
                'no-auxiliary-targets',
                'no-root-value-blend',
                'true-starts-only',
                'fixed-search-budgets',
                'no-tree-retention',
                'no-mixed-search',
                'replay-ratio-8',
                'constant-learning-rate-007',
            )
        )
    )
