from __future__ import annotations

from pathlib import Path

import pytest
from markdown_it import MarkdownIt
from tools.render_technical_report import bibliography_tex, caption_tex, escape_tex, inline_tex, section_tex, table_tex


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


def test_investigations_use_numeric_section_hierarchy() -> None:
    search = Path('04a-search.md')
    assert r'\section{Research investigations}' in section_tex(
        '4.1. Search', appendix=False, level=1, appendix_letter='', source=search
    )
    assert r'\subsection{Search}' in section_tex(
        '4.1. Search', appendix=False, level=1, appendix_letter='', source=search
    )
    assert section_tex('Search budgets', appendix=False, level=2, appendix_letter='', source=search) == (
        '\\subsubsection{Search budgets}\n'
    )


def test_figure_caption_uses_latex_counter_instead_of_manual_number() -> None:
    assert caption_tex('Figure 7.2: Strength across budgets.', 'strength') == (
        r'\caption{Strength across budgets.}\label{fig:strength}'
    )


def test_appendix_table_has_caption_without_upscaling_type() -> None:
    output = table_tex(
        [['Heading', 'Value'], ['Item', '1']],
        source=Path('appendix-c-supporting-comparisons.md'),
        table_number=2,
        appendix=True,
    )
    assert r'\caption{Parallel-search strength and wall time}' in output
    assert r'\centering\normalsize' in output
    assert r'\setlength{\tabcolsep}{9pt}' in output
    assert r'\resizebox' not in output


def test_main_result_table_has_caption() -> None:
    output = table_tex(
        [['Searches', 'Elo'], ['100,000', '3,251']],
        source=Path('02-methodology-and-evidence.md'),
        table_number=1,
        appendix=False,
    )
    assert r'\begin{table*}[!t]' in output
    assert r'\caption{Final checkpoint against fixed-node Stockfish 13}' in output
    assert r'\label{tab:02-methodology-and-evidence-1}' in output
    assert output.endswith('\\FloatBarrier\n')
