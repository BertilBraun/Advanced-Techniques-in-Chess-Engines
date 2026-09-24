from __future__ import annotations

import pytest
from markdown_it import MarkdownIt
from tools.render_technical_report import bibliography_tex, escape_tex, inline_tex


@pytest.mark.parametrize(
    ('source', 'expected'),
    [
        ('KataGo [7]', r'KataGo \cite{ref7}'),
        ('Elo [3,065, 3,163]', 'Elo [3,065, 3,163]'),
        ('50% and 7×7', r'50\% and 7\ensuremath{\times}7'),
    ],
)
def test_report_text_escapes_citations_without_changing_intervals(source: str, expected: str) -> None:
    assert escape_tex(source) == expected


def test_bibliography_has_clickable_numbered_targets() -> None:
    bibliography = bibliography_tex()
    assert bibliography.count(r'\bibitem{ref') == 12
    assert r'\bibitem{ref7}' in bibliography
    assert r'\href{https://github.com/lightvector/KataGo/blob/v1.17.1/SelfplayTraining.md}' in bibliography


def test_report_prose_rejects_external_markdown_links() -> None:
    markdown = MarkdownIt('commonmark')
    paragraph = markdown.parse('[external](https://example.com)')[1]
    with pytest.raises(ValueError, match='bibliography citations'):
        inline_tex(paragraph.children or [])
