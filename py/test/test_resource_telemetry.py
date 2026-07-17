from unittest.mock import Mock

from src.experiment.resource_telemetry import (
    parse_nvidia_smi_output,
    process_tree_open_file_counts,
)


def test_parse_nvidia_smi_output() -> None:
    output = (
        '0, NVIDIA GeForce RTX 4070, 87, 8021, 12282, 65, 172.50\n'
        '1, NVIDIA GeForce RTX 4070, 91, 7912, 12282, 66, 175.25\n'
    )

    samples = parse_nvidia_smi_output(output)

    assert len(samples) == 2
    assert samples[0].index == 0
    assert samples[0].utilization_percent == 87
    assert samples[0].memory_total_mib == 12282
    assert samples[1].power_watts == 175.25


def test_process_tree_open_file_counts_tracks_maximum_and_total() -> None:
    first_child = Mock()
    second_child = Mock()
    parent = Mock()
    parent.children.return_value = [first_child, second_child]
    parent.num_handles.return_value = 12
    parent.num_fds.return_value = 12
    first_child.num_handles.return_value = 40
    first_child.num_fds.return_value = 40
    second_child.num_handles.return_value = 7
    second_child.num_fds.return_value = 7

    maximum_count, total_count = process_tree_open_file_counts(parent)

    assert maximum_count == 40
    assert total_count == 59
