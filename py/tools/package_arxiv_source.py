"""Package the rendered report and its referenced PDF figures for arXiv."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GRAPHICS = re.compile(r'(\\includegraphics\[[^\]]*\]\{)([^}]+)(\})')
PROCESSING_INSTRUCTIONS = """{
  "process": {"compiler": "xelatex"},
  "sources": [{"filename": "main.tex", "usage": "toplevel"}]
}
"""


def package_source(source: Path, output: Path) -> None:
    latex = source.read_text(encoding='utf-8')
    figures: set[Path] = set()

    def localize_figure(match: re.Match[str]) -> str:
        figure = Path(match.group(2))
        if not figure.is_absolute():
            figure = source.parent / figure
        figure = figure.resolve()
        if not figure.is_relative_to(source.parent.resolve() / 'figures') or figure.suffix != '.pdf':
            raise ValueError(f'Expected a rendered PDF figure in the build directory: {figure}')
        if not figure.is_file():
            raise ValueError(f'Missing figure: {figure}')
        figures.add(figure)
        return match.group(1) + 'figures/' + figure.name + match.group(3)

    latex = GRAPHICS.sub(localize_figure, latex)
    if not figures:
        raise ValueError('Report contains no figures.')
    if len({figure.name for figure in figures}) != len(figures):
        raise ValueError('Figure basenames must be unique.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'w', compression=ZIP_DEFLATED) as archive:
        archive.writestr('main.tex', latex)
        archive.writestr('00README.json', PROCESSING_INSTRUCTIONS)
        for figure in sorted(figures):
            archive.write(figure, 'figures/' + figure.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source',
        type=Path,
        default=REPOSITORY_ROOT / 'tmp/pdfs/latex-build/technical-report-review.tex',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=REPOSITORY_ROOT / 'output/arxiv/engineering-efficient-self-play-chess-source.zip',
    )
    arguments = parser.parse_args()
    package_source(arguments.source.resolve(), arguments.output.resolve())
    print(arguments.output.resolve())


if __name__ == '__main__':
    main()
