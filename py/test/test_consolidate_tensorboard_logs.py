from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboardX import SummaryWriter

from tools.consolidate_tensorboard_logs import TensorboardLogConsolidator, consolidate_once


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


def custom_scalar_layout(output_root: Path) -> layout_pb2.Layout:
    accumulator = EventAccumulator(str(output_root))
    accumulator.Reload()
    layout_event = accumulator.Tensors('custom_scalars__config__')[-1]
    return layout_pb2.Layout.FromString(layout_event.tensor_proto.string_val[0])


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


def test_consolidation_accepts_one_run_directory_as_source(tmp_path: Path) -> None:
    source_run_directory = tmp_path / 'source-run'
    output_root = tmp_path / 'output'
    for series_name, value in (('wins', 20.0), ('draws', 70.0), ('losses', 10.0)):
        write_scalar(
            source_run_directory / 'coordinator',
            f'evaluation/stockfish-level-0/{series_name}',
            1_200,
            value,
            wall_time=10.0,
        )
    write_scalar(
        source_run_directory / 'self_play' / '100',
        'mcts/average_search_depth',
        1,
        4.0,
        wall_time=10.0,
    )

    manifest = consolidate_once(
        source_run_directory,
        output_root,
        time_series_view=True,
        single_run_source=True,
    )

    for series_name, expected_value in (('wins', 20.0), ('draws', 70.0), ('losses', 10.0)):
        run_directory = output_root / 'coordinator' / 'evaluation' / 'stockfish-level-0' / series_name
        assert scalar_values(run_directory, 'coordinator/evaluation/stockfish-level-0') == (expected_value,)
    assert manifest.selected_event_file_count == 4
    assert manifest.representative_self_play_processes[0].process_id == 100


def test_evaluation_outcome_overlays_preserve_other_summaries_in_root_run(tmp_path: Path) -> None:
    source_run_directory = tmp_path / 'source-run'
    output_root = tmp_path / 'output'
    write_scalar(source_run_directory / 'coordinator', 'training/total_loss', 1, 2.0, wall_time=10.0)
    write_scalar(
        source_run_directory / 'coordinator',
        'evaluation/stockfish-level-0/wins',
        1_200,
        20.0,
        wall_time=10.0,
    )
    write_scalar(
        source_run_directory / 'coordinator',
        'evaluation/stockfish-level-0/score',
        1_200,
        0.2,
        wall_time=10.0,
    )

    consolidate_once(
        source_run_directory,
        output_root,
        single_run_source=True,
        evaluation_outcome_overlays=True,
    )

    assert scalar_values(output_root, 'coordinator/training/total_loss') == (2.0,)
    assert scalar_values(output_root, 'coordinator/evaluation/stockfish-level-0/score') == pytest.approx((0.2,))
    outcome_run = output_root / 'coordinator' / 'evaluation' / 'stockfish-level-0' / 'wins'
    assert scalar_values(outcome_run, 'coordinator/evaluation/stockfish-level-0') == (20.0,)


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


def test_consolidation_preserves_evaluation_boundary_summary(tmp_path: Path) -> None:
    source_root = tmp_path / 'source'
    output_root = tmp_path / 'output'
    writer = SummaryWriter(str(source_root / 'run_0' / 'coordinator'))
    writer.add_text('evaluation/summaries', '| Evaluation | Score |', global_step=1_200)
    writer.close()

    manifest = consolidate_once(source_root, output_root)

    accumulator = EventAccumulator(str(output_root))
    accumulator.Reload()
    events = accumulator.Tensors('coordinator/evaluation/summaries/text_summary')
    assert len(events) == 1
    assert manifest.excluded_text_summary_count == 0


def test_custom_scalar_layout_groups_discovered_evaluation_definitions(tmp_path: Path) -> None:
    source_root = tmp_path / 'source'
    output_root = tmp_path / 'output'
    evaluation_directory = source_root / 'run_0' / 'coordinator'
    for metric, value in (
        ('wins', 12.0),
        ('draws', 0.0),
        ('losses', 88.0),
        ('score', 0.12),
    ):
        write_scalar(evaluation_directory, f'evaluation/katago-16/{metric}', 1_200, value, wall_time=10.0)
    for metric, value in (
        ('first_player_score', 0.0),
        ('second_player_score', 0.24),
        ('duration_seconds', 80.0),
    ):
        write_scalar(
            evaluation_directory,
            f'evaluation_metadata/katago-16/{metric}',
            1_200,
            value,
            wall_time=10.0,
        )
    write_scalar(
        evaluation_directory,
        'evaluation_metadata/previous-20m/duration_seconds',
        1_200,
        70.0,
        wall_time=10.0,
    )

    consolidate_once(source_root, output_root)

    layout = custom_scalar_layout(output_root)
    match_category = next(category for category in layout.category if category.title == 'Evaluation matches')
    assert tuple(chart.title for chart in match_category.chart) == ('katago-16 W/D/L', 'katago-16 score')
    assert tuple(match_category.chart[0].multiline.tag) == (
        'coordinator/evaluation/katago-16/wins',
        'coordinator/evaluation/katago-16/draws',
        'coordinator/evaluation/katago-16/losses',
    )
    assert tuple(match_category.chart[1].multiline.tag) == ('coordinator/evaluation/katago-16/score',)
    metadata_category = next(category for category in layout.category if category.title == 'Evaluation metadata')
    assert tuple(metadata_category.chart[0].multiline.tag) == (
        'coordinator/evaluation_metadata/katago-16/first_player_score',
        'coordinator/evaluation_metadata/katago-16/second_player_score',
    )
    assert tuple(metadata_category.chart[1].multiline.tag) == (
        'coordinator/evaluation_metadata/katago-16/duration_seconds',
        'coordinator/evaluation_metadata/previous-20m/duration_seconds',
    )


def test_custom_scalar_layout_compacts_discovered_fixed_dataset_metrics(tmp_path: Path) -> None:
    source_root = tmp_path / 'source'
    output_root = tmp_path / 'output'
    evaluation_directory = source_root / 'run_0' / 'evaluation_fixed_dataset'
    write_scalar(
        evaluation_directory,
        'evaluation/fixed-dataset/top_action_accuracy',
        1_200,
        0.2,
        wall_time=10.0,
    )
    write_scalar(
        evaluation_directory,
        'evaluation/fixed-dataset/policy_cross_entropy',
        1_200,
        3.1,
        wall_time=10.0,
    )
    write_scalar(
        evaluation_directory,
        'evaluation/fixed-dataset/duration_seconds',
        1_200,
        5.0,
        wall_time=10.0,
    )

    consolidate_once(source_root, output_root)

    layout = custom_scalar_layout(output_root)
    dataset_category = next(category for category in layout.category if category.title == 'Evaluation datasets')
    assert tuple(chart.title for chart in dataset_category.chart) == (
        'Top-action accuracy',
        'Policy cross-entropy',
    )
    assert tuple(dataset_category.chart[0].multiline.tag) == ('evaluation/fixed-dataset/top_action_accuracy',)
    assert tuple(dataset_category.chart[1].multiline.tag) == ('evaluation/fixed-dataset/policy_cross_entropy',)


def test_watched_consolidation_updates_layout_for_new_definition(tmp_path: Path) -> None:
    source_root = tmp_path / 'source'
    output_root = tmp_path / 'output'
    write_scalar(
        source_root / 'run_0' / 'evaluation_katago_16',
        'evaluation/katago-16/duration_seconds',
        1_200,
        80.0,
        wall_time=10.0,
    )
    consolidator = TensorboardLogConsolidator(source_root, output_root)
    try:
        consolidator.scan()
        write_scalar(
            source_root / 'run_0' / 'evaluation_same_time_baseline',
            'evaluation/same-time-baseline/duration_seconds',
            1_200,
            90.0,
            wall_time=20.0,
        )
        consolidator.scan()
    finally:
        consolidator.close()

    layout = custom_scalar_layout(output_root)
    metadata_category = next(category for category in layout.category if category.title == 'Evaluation metadata')
    assert tuple(metadata_category.chart[0].multiline.tag) == (
        'evaluation/katago-16/duration_seconds',
        'evaluation/same-time-baseline/duration_seconds',
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


def test_time_series_view_gives_standalone_metrics_semantic_colored_runs(tmp_path: Path) -> None:
    source_root = tmp_path / 'source'
    output_root = tmp_path / 'output'
    write_scalar(
        source_root / 'run_0' / 'gpu_usage',
        'gpu/0/load',
        1,
        75.0,
        wall_time=10.0,
    )

    consolidate_once(source_root, output_root, time_series_view=True)

    run_directory = output_root / 'system' / 'gpu' / 'gpu' / '0' / 'load'
    assert scalar_values(run_directory, 'system/gpu/gpu/0/load') == (75.0,)
    assert not (output_root / 'other_metrics').exists()
