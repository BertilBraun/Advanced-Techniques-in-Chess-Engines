from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest
from tools.package_arxiv_source import package_source


def test_package_contains_only_source_instructions_and_used_figures(tmp_path: Path) -> None:
    figures = tmp_path / 'figures'
    figures.mkdir()
    figure = figures / 'used.pdf'
    figure.write_bytes(b'figure')
    (figures / 'unused.pdf').write_bytes(b'unused')
    source = tmp_path / 'report.tex'
    source.write_text(r'\includegraphics[width=1in]{' + figure.as_posix() + '}', encoding='utf-8')
    output = tmp_path / 'submission.zip'
    package_source(source, output)
    with ZipFile(output) as archive:
        assert set(archive.namelist()) == {'main.tex', '00README.json', 'figures/used.pdf'}
        assert archive.read('main.tex').decode() == r'\includegraphics[width=1in]{figures/used.pdf}'
        assert 'xelatex' in archive.read('00README.json').decode()


def test_package_rejects_figure_outside_build_directory(tmp_path: Path) -> None:
    source = tmp_path / 'report.tex'
    source.write_text(r'\includegraphics[width=1in]{../private.pdf}', encoding='utf-8')
    with pytest.raises(ValueError, match='Expected a rendered PDF'):
        package_source(source, tmp_path / 'submission.zip')
