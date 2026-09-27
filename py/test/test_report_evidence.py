from __future__ import annotations

import csv
from pathlib import Path

import pytest
from tools.check_report_links import check_link, heading_slug

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_PATH = REPOSITORY_ROOT / 'documentation/evidence/final-chess-20260923/training-trajectory.csv'
PLATEAU_PATH = REPOSITORY_ROOT / 'documentation/evidence/final-chess-20260923/plateau-comparison.csv'


@pytest.mark.parametrize(
    ('heading', 'expected'),
    (
        ('Training volume & throughput', 'training-volume-throughput'),
        ('`TensorRT` refit', 'tensorrt-refit'),
        ('[Final result](../results/final-chess-run.md)', 'final-result'),
    ),
)
def test_heading_slug(heading: str, expected: str) -> None:
    assert heading_slug(heading) == expected


def test_report_link_checks_local_anchor(tmp_path: Path) -> None:
    source = tmp_path / 'source.md'
    target = tmp_path / 'target.md'
    source.write_text('See [the target](target.md#target-heading).', encoding='utf-8')
    target.write_text('# Target heading\n', encoding='utf-8')
    assert check_link(source, 1, 'target.md#target-heading') is None
    assert check_link(source, 1, 'target.md#missing-heading') is not None


def test_final_training_trajectory_reaches_selected_checkpoint() -> None:
    with TRAJECTORY_PATH.open(encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 817
    assert [int(row['tensorboard_step']) for row in rows] == list(range(1, 818))
    assert sum(int(row['completed_games']) for row in rows) == 3_249_647
    assert int(rows[-1]['optimizer_steps']) == 408_500
    assert int(rows[-1]['consumed_presentations']) == 836_608_000
    assert int(rows[-1]['replay_live_rows']) == 16_000_000


def test_matched_plateau_and_clean_cutoff_sensitivity() -> None:
    with PLATEAU_PATH.open(encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    assert [row['window'] for row in rows] == ['original_retrospective', 'publication_cutoff_sensitivity']
    assert [int(row['previous_count']) for row in rows] == [59, 15]
    assert [int(row['final_count']) for row in rows] == [86, 37]
    assert [float(row['matched_difference']) for row in rows] == pytest.approx([74.1271, 75.7593])
