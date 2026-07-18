from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from tools.consolidate_tensorboard_logs import consolidate_once


def write_scalar(log_directory: Path, tag: str, step: int, value: float, wall_time: float) -> None:
    writer = EventFileWriter(str(log_directory), max_queue_size=1)
    writer.add_event(
        event_pb2.Event(
            wall_time=wall_time,
            step=step,
            summary=summary_pb2.Summary(
                value=(summary_pb2.Summary.Value(tag=tag, simple_value=value),),
            ),
        )
    )
    writer.close()


def scalar_values(output_root: Path, tag: str) -> tuple[float, ...]:
    accumulator = EventAccumulator(str(output_root))
    accumulator.Reload()
    return tuple(event.value for event in accumulator.Scalars(tag))


def test_consolidation_merges_run_fragments_and_keeps_newest_duplicate(tmp_path: Path) -> None:
    source_root = tmp_path / 'source'
    output_root = tmp_path / 'output'
    write_scalar(source_root / 'run_0' / 'trainer', 'training/loss', 4, 2.0, wall_time=10.0)
    write_scalar(source_root / 'run_1' / 'trainer', 'training/loss', 4, 1.5, wall_time=20.0)
    write_scalar(source_root / 'run_1' / 'trainer', 'training/loss', 5, 1.0, wall_time=21.0)

    manifest = consolidate_once(source_root, output_root)

    assert scalar_values(output_root, 'training/loss') == (1.5, 1.0)
    assert manifest.unique_summary_count == 2
    assert manifest.replaced_summary_count == 1
    assert manifest.tags[0].minimum_step == 4
    assert manifest.tags[0].maximum_step == 5


def test_consolidation_selects_one_self_play_process_per_fragment(tmp_path: Path) -> None:
    source_root = tmp_path / 'source'
    output_root = tmp_path / 'output'
    write_scalar(
        source_root / 'run_0' / 'self_play' / '100',
        'mcts/average_search_depth',
        1,
        4.0,
        wall_time=10.0,
    )
    write_scalar(
        source_root / 'run_0' / 'self_play' / '101',
        'mcts/average_search_depth',
        1,
        9.0,
        wall_time=10.0,
    )
    write_scalar(
        source_root / 'run_1' / 'self_play' / '200',
        'mcts/average_search_depth',
        2,
        5.0,
        wall_time=20.0,
    )

    manifest = consolidate_once(source_root, output_root)

    assert scalar_values(output_root, 'self_play/mcts/average_search_depth') == (4.0, 5.0)
    assert manifest.skipped_self_play_event_file_count == 1
    assert tuple(item.process_id for item in manifest.representative_self_play_processes) == (100, 200)


def test_consolidation_namespaces_usage_metrics(tmp_path: Path) -> None:
    source_root = tmp_path / 'source'
    output_root = tmp_path / 'output'
    write_scalar(
        source_root / 'run_0' / 'cpu_usage_trainer' / '100',
        'usage/cpu_percent',
        1,
        50.0,
        wall_time=10.0,
    )
    write_scalar(
        source_root / 'run_0' / 'cpu_usage_self_play' / '101',
        'usage/cpu_percent',
        1,
        30.0,
        wall_time=10.0,
    )

    consolidate_once(source_root, output_root)

    assert scalar_values(output_root, 'system/cpu_trainer/usage/cpu_percent') == (50.0,)
    assert scalar_values(output_root, 'system/cpu_self_play/usage/cpu_percent') == (30.0,)


def test_consolidation_preserves_add_scalars_child_series(tmp_path: Path) -> None:
    source_root = tmp_path / 'source'
    output_root = tmp_path / 'output'
    write_scalar(
        source_root / 'run_0' / 'evaluation_dataset' / 'evaluation' / 'policy_accuracy' / '1',
        'evaluation/policy_accuracy',
        4,
        0.2,
        wall_time=10.0,
    )
    write_scalar(
        source_root / 'run_0' / 'evaluation_dataset' / 'evaluation' / 'policy_accuracy' / '5',
        'evaluation/policy_accuracy',
        4,
        0.6,
        wall_time=10.0,
    )

    consolidate_once(source_root, output_root)

    assert scalar_values(output_root, 'evaluation/policy_accuracy/1') == pytest.approx((0.2,))
    assert scalar_values(output_root, 'evaluation/policy_accuracy/5') == pytest.approx((0.6,))


def test_custom_scalar_layout_groups_wins_draws_and_losses(tmp_path: Path) -> None:
    source_root = tmp_path / 'source'
    output_root = tmp_path / 'output'
    write_scalar(source_root / 'run_0' / 'trainer', 'train/total_loss', 1, 1.0, wall_time=10.0)

    consolidate_once(source_root, output_root)

    accumulator = EventAccumulator(str(output_root))
    accumulator.Reload()
    layout_event = accumulator.Tensors('custom_scalars__config__')[0]
    layout = layout_pb2.Layout.FromString(layout_event.tensor_proto.string_val[0])
    wdl_category = next(category for category in layout.category if category.title == 'Evaluation W/D/L')
    reference_chart = next(chart for chart in wdl_category.chart if chart.title == 'Reference model')
    assert tuple(reference_chart.multiline.tag) == (
        'evaluation/vs_reference_model/wins',
        'evaluation/vs_reference_model/draws',
        'evaluation/vs_reference_model/losses',
    )


def test_time_series_view_overlays_evaluation_series_as_colored_runs(tmp_path: Path) -> None:
    source_root = tmp_path / 'source'
    output_root = tmp_path / 'output'
    for series_name, value in (('wins', 20.0), ('draws', 70.0), ('losses', 10.0), ('score', 55.0)):
        write_scalar(
            source_root
            / 'run_0'
            / 'evaluation_vs_stockfish_level_0'
            / 'evaluation'
            / 'vs_stockfish_level_0'
            / series_name,
            'evaluation/vs_stockfish_level_0',
            4,
            value,
            wall_time=10.0,
        )

    consolidate_once(source_root, output_root, time_series_view=True)

    for series_name, expected_value in (('wins', 20.0), ('draws', 70.0), ('losses', 10.0), ('score', 55.0)):
        run_directory = output_root / 'evaluation' / 'vs_stockfish_level_0' / series_name
        assert scalar_values(run_directory, 'evaluation/vs_stockfish_level_0') == (expected_value,)
