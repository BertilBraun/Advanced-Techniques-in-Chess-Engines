from __future__ import annotations

from pathlib import Path

import pytest
from markdown_it import MarkdownIt
from tools.render_technical_report import (
    REPORT_ROOT,
    advance_main_figures,
    bibliography_tex,
    caption_tex,
    escape_tex,
    figure_width_points,
    inline_tex,
    markdown_tex,
    section_tex,
    table_tex,
)


@pytest.mark.parametrize(
    'body',
    [
        '<style>.label {font-size: 12px;}</style><text class="label">Label</text>',
        '<text font-size="12">Label</text>',
        '<g id="text_1"><g transform="translate(1 2) scale(0.12 -0.12)"><path/></g></g>',
    ],
)
def test_figure_width_caps_largest_label_at_ten_points(tmp_path: Path, body: str) -> None:
    source = tmp_path / 'figure.svg'
    source.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 180">' + body + '</svg>', encoding='utf-8'
    )
    assert figure_width_points(source) == pytest.approx(300)


def test_main_figures_are_queued_early_without_changing_their_order() -> None:
    first = '\\begin{figure*}[t]\nFIRST\n\\end{figure*}\n'
    second = '\\begin{figure*}[t]\nSECOND\n\\end{figure*}\n'
    paragraph = ' '.join(['prose'] * 300) + '\n\n'
    latex = paragraph * 3 + first + paragraph * 3 + second
    adjusted = advance_main_figures(latex)
    assert adjusted.index(first) < latex.index(first)
    assert adjusted.index(first) < adjusted.index(second)
    assert adjusted.replace(first, '').replace(second, '') == latex.replace(first, '').replace(second, '')


def test_system_diagrams_respect_float_limit_and_loss_has_two_forms(tmp_path: Path) -> None:
    output = markdown_tex(REPORT_ROOT / '03-system-and-methods.md', tmp_path)
    assert output.count(r'\begin{figure*}[t]') == 2
    assert r'\begin{figure*}[!t]' not in output
    assert r'\operatorname{CE}(\pi,p) + \operatorname{CE}(z,v)' in output
    assert r'&= -\sum_a \pi_a\log p_a' in output


def test_display_math_preserves_latex_equation(tmp_path: Path) -> None:
    source = tmp_path / 'equations.md'
    source.write_text('```math\na = \\frac{b}{c}\n```\n', encoding='utf-8')
    assert markdown_tex(source, tmp_path) == '\\begin{equation}\na = \\frac{b}{c}\n\\end{equation}\n'


def test_input_plane_table_wraps_descriptions_at_page_width() -> None:
    output = table_tex(
        [['Planes', 'Feature', 'Encoding'], ['0', 'Own pawns', 'Binary mask']],
        source=Path('appendix-d-reproducibility.md'),
        table_number=1,
        appendix=True,
    )
    assert r'\begin{tabularx}{\textwidth}{@{}l X l@{}}' in output
    assert r'\end{tabularx}' in output
    assert r'\label{tab:appendix-d-reproducibility-1}' in output


def test_training_evidence_is_in_main_text_without_duplicate_appendix_figure() -> None:
    training = (REPORT_ROOT / '06-final-chess-recipe.md').read_text(encoding='utf-8')
    diagnostics = (REPORT_ROOT / 'appendix-a-training-diagnostics.md').read_text(encoding='utf-8')
    assert 'final-training-volume-and-throughput-paper.svg' in training
    assert 'final-training-volume-and-throughput-paper.svg' not in diagnostics
    assert 'appendix-training-stages.svg' not in diagnostics
    assert '3.25 million' in training


def test_main_results_tables_and_architecture_sweeps_render(tmp_path: Path) -> None:
    for filename, count in (
        ('05-systems-optimization.md', 1),
        ('06-final-chess-recipe.md', 1),
        ('07-final-run-results.md', 2),
        ('appendix-c-supporting-comparisons.md', 2),
    ):
        output = markdown_tex(REPORT_ROOT / filename, tmp_path, appendix=filename.startswith('appendix-'))
        assert output.count(r'\begin{table') == count


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


def test_roadmap_has_clickable_internal_section_links() -> None:
    paragraph = MarkdownIt('commonmark').parse('[Chapter 3](03-system-and-methods.md)')[1]
    assert inline_tex(paragraph.children or []) == r'\hyperref[sec:03-system-and-methods]{Chapter 3}'


@pytest.mark.parametrize(
    ('filename', 'letter'),
    [('appendix-c-supporting-comparisons.md', 'C'), ('appendix-d-reproducibility.md', 'D')],
)
def test_appendix_links_target_the_pdf_appendix(filename: str, letter: str) -> None:
    paragraph = MarkdownIt('commonmark').parse(f'[Appendix {letter}]({filename})')[1]
    assert inline_tex(paragraph.children or []) == rf'\hyperref[app:{letter}]{{Appendix {letter}}}'


def test_investigations_use_numeric_section_hierarchy() -> None:
    search = Path('04a-search.md')
    assert r'\section{Research investigations}' in section_tex(
        '4. Research investigations',
        appendix=False,
        level=1,
        appendix_letter='',
        source=Path('04-research-investigations.md'),
    )
    assert r'\section{' not in section_tex('4.1. Search', appendix=False, level=1, appendix_letter='', source=search)
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
    assert r'\caption{Depth and width: throughput ratios across batch sizes}' in output
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
    assert r'\FloatBarrier' not in output
