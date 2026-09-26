from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from pydantic import ValidationError
from tools import render_report_figures as renderer


@pytest.fixture
def diagnostics() -> renderer.Diagnostics:
    return renderer.Diagnostics.model_validate_json(
        (renderer.EVIDENCE / 'training-diagnostics.json').read_text(encoding='utf-8')
    )


@pytest.mark.parametrize(
    'metric',
    [
        renderer.Metric.NEXT_POLICY,
        renderer.Metric.REMAINING_LENGTH,
        renderer.Metric.GRADIENT,
        renderer.Metric.VISITS,
        renderer.Metric.REPLAY_AGE,
    ],
)
def test_diagnostics_match_selected_training_steps(diagnostics: renderer.Diagnostics, metric: renderer.Metric) -> None:
    assert [sample.optimizer_steps for sample in diagnostics.samples(metric)] == list(range(500, 408_501, 500))
    assert diagnostics.selected_optimizer_steps == 408_500
    assert diagnostics.trajectory_sha256 == renderer.digest(renderer.EVIDENCE / 'training-trajectory.csv')


def test_diagnostics_retain_safety_excursions(diagnostics: renderer.Diagnostics) -> None:
    assert any(sample.value > 0.025 for sample in diagnostics.samples(renderer.Metric.UPPER_BOUND))
    assert diagnostics.samples(renderer.Metric.REPLAY_AGE)[-1].value > 3600


def test_diagnostic_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        renderer.Sample.model_validate({'optimizer_steps': 500, 'value': 0.1, 'wrong_axis': 1})


def test_archive_hash_is_required(tmp_path: Path) -> None:
    wrong_archive = tmp_path / 'wrong.tgz'
    wrong_archive.write_bytes(b'not the frozen archive')
    with pytest.raises(ValueError, match='frozen evidence hash'):
        renderer.extract_diagnostics(wrong_archive, renderer.EVIDENCE / 'training-trajectory.csv')


def test_ladder_stops_at_two_and_a_half_days(monkeypatch: pytest.MonkeyPatch) -> None:
    figures: list[Figure] = []

    def capture(figure: Figure, path: Path) -> None:
        figures.append(figure)

    monkeypatch.setattr(renderer, 'save_figure', capture)
    renderer.render_ladder(renderer.EVIDENCE / 'ladder-elo-export.json')
    assert len(figures) == 1
    axes = figures[0].axes[0]
    assert axes.get_xlim() == pytest.approx((0, 2.5))
    for curve in axes.lines:
        assert max(curve.get_xdata()) == pytest.approx(2.5)
    plt.close(figures[0])


def test_resignation_uses_three_separate_axes(
    monkeypatch: pytest.MonkeyPatch, diagnostics: renderer.Diagnostics
) -> None:
    figures: list[Figure] = []

    def capture(figure: Figure, path: Path) -> None:
        figures.append(figure)

    monkeypatch.setattr(renderer, 'save_figure', capture)
    renderer.render_resignation(diagnostics)
    figure = figures[0]
    assert len(figure.axes) == 3
    assert [axes.get_ylabel() for axes in figure.axes] == [
        'Resignation threshold',
        'False non-loss (%)',
        'Mean saved plies',
    ]
    assert all(axes.get_title() == '' for axes in figure.axes)
    assert figure.axes[1].get_ylim()[1] > 100 * max(
        sample.value
        for sample in diagnostics.samples(renderer.Metric.UPPER_BOUND)
        if sample.optimizer_steps
        in {point.optimizer_steps for point in diagnostics.samples(renderer.Metric.SAFE) if point.value == 1}
    )
    plt.close(figure)
