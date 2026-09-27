from __future__ import annotations

from pathlib import Path

import pytest
from tools.render_archived_tensorboard_figures import FIGURES, parse_arguments


def test_archive_renderer_keeps_its_output_separate_from_publication_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'sys.argv', ['render_archived_tensorboard_figures', '--tensorboard-root', 'evidence/tensorboard']
    )

    arguments = parse_arguments()

    assert arguments.output_directory == Path('documentation/history/archive-diagnostic-figures')
    assert arguments.tensorboard_root == Path('evidence/tensorboard')
    assert len(FIGURES) == 9
