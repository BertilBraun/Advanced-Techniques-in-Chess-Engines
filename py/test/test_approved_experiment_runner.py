from pathlib import Path

from run_approved_experiment import approval_path


def test_approval_path_is_selected_from_authored_configuration_name(tmp_path: Path) -> None:
    approval_directory = tmp_path / 'approvals'

    assert approval_path(approval_directory, Path('configs/10-random-opening-12.yaml')) == (
        approval_directory / '10-random-opening-12.json'
    )
