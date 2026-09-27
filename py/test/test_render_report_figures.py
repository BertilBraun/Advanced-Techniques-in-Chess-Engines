from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
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
    assert [axes.get_title() for axes in figure.axes] == [
        'Resignation threshold',
        'False non-loss (%)',
        'Mean saved plies',
    ]
    assert all(axes.get_xlabel() == '' for axes in figure.axes)
    assert figure.get_supxlabel() == ''
    assert len({round(axes.get_position().y0, 6) for axes in figure.axes}) == 1
    assert all(axes.title.get_fontsize() == 10 for axes in figure.axes)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer_backend = canvas.get_renderer()
    tick_bottom = min(label.get_window_extent(renderer_backend).y0 for label in figure.axes[1].get_xticklabels())
    legend_top = figure.legends[0].get_window_extent(renderer_backend).y1
    assert 0 < (tick_bottom - legend_top) / figure.dpi < 0.12
    assert figure.axes[1].get_ylim()[1] > 100 * max(
        sample.value
        for sample in diagnostics.samples(renderer.Metric.UPPER_BOUND)
        if sample.optimizer_steps
        in {point.optimizer_steps for point in diagnostics.samples(renderer.Metric.SAFE) if point.value == 1}
    )
    plt.close(figure)


def test_saved_plot_crops_unused_canvas(tmp_path: Path) -> None:
    figure = plt.figure(figsize=(6, 4))
    axes = figure.add_axes((0.3, 0.3, 0.4, 0.4))
    axes.plot([0, 1], [0, 1])
    path = tmp_path / 'cropped.svg'
    renderer.save_figure(figure, path)
    view_box = [float(value) for value in ElementTree.parse(path).getroot().attrib['viewBox'].split()]
    assert view_box[2] < 6 * 72 * 0.7
    assert view_box[3] < 4 * 72 * 0.7
