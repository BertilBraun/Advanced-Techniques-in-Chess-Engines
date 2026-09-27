"""Build the Markdown report with the Voice-Light two-column LaTeX layout."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import xml.etree.ElementTree as element_tree
from pathlib import Path
from urllib.parse import urlparse

from markdown_it import MarkdownIt
from markdown_it.token import Token
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPOSITORY_ROOT / 'documentation' / 'report'
SOURCE_FILES = (
    '01-motivation-and-scope.md',
    '02-methodology-and-evidence.md',
    '03-system-and-methods.md',
    '04-research-investigations.md',
    '04a-search.md',
    '04b-data-and-replay.md',
    '04c-networks-and-training.md',
    '05-systems-optimization.md',
    '05a-three-failures.md',
    '06-final-chess-recipe.md',
    '07-final-run-results.md',
    '08-limitations.md',
    '10-conclusion.md',
)
APPENDIX_FILES = (
    'appendix-a-training-diagnostics.md',
    'appendix-b-evaluation-tables.md',
    'appendix-c-supporting-comparisons.md',
    'appendix-d-reproducibility.md',
)
PREAMBLE = r"""\documentclass[10pt,twocolumn]{article}
\usepackage[a4paper,top=18mm,bottom=20mm,left=16mm,right=16mm,columnsep=7mm]{geometry}
\usepackage{amsmath}
\usepackage{newtxtext,newtxmath}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{hyperref}
\usepackage{xurl}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{caption}
\usepackage{float}
\usepackage{placeins}
\usepackage{needspace}
\usepackage{balance}
\definecolor{vlteal}{HTML}{16615A}
\definecolor{vltealdark}{HTML}{0E4742}
\hypersetup{colorlinks=true,linkcolor=vltealdark,citecolor=vltealdark,urlcolor=vlteal,
  pdftitle={Engineering Efficient Self-Play Chess},pdfauthor={Bertil Braun}}
\titleformat{\section}{\large\bfseries}{\thesection}{0.55em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.5em}{}
\titlespacing*{\section}{0pt}{1.15em}{0.45em}
\titlespacing*{\subsection}{0pt}{0.9em}{0.3em}
\setlength{\parindent}{1em}
\setlength{\parskip}{0pt}
\widowpenalty=10000
\clubpenalty=10000
\raggedbottom
\setlength{\columnsep}{7mm}
\setcounter{dbltopnumber}{2}
\setlist[itemize]{leftmargin=1.25em,itemsep=0.08em,topsep=0.25em}
\setlist[enumerate]{leftmargin=1.35em,itemsep=0.12em,topsep=0.25em}
\captionsetup{font=small,labelfont=bf}
\title{\textbf{Engineering Efficient Self-Play Chess: Search, Replay, and\\Throughput Under Limited Compute}}
\author{Bertil Braun\\
  \small \href{mailto:contact@bertil-braun.de}{contact@bertil-braun.de}}
\date{}
\begin{document}
\twocolumn[{
\begin{@twocolumnfalse}
\maketitle
\vspace{-1.8em}
\begin{abstract}
"""
POST_ABSTRACT = r"""\end{abstract}
\vspace{0.45em}
\noindent\textbf{Keywords:} AlphaZero, chess, self-play, Monte Carlo tree search, replay,
GPU inference, quantization, model growth, fixed-node evaluation
\vspace{1.0em}
\end{@twocolumnfalse}
}]
"""
SPECIAL_CHARACTERS = {
    '\\': r'\textbackslash{}',
    '{': r'\{',
    '}': r'\}',
    '#': r'\#',
    '$': r'\$',
    '%': r'\%',
    '&': r'\&',
    '_': r'\_',
    '^': r'\textasciicircum{}',
    '~': r'\textasciitilde{}',
    '±': r'\ensuremath{\pm}',
    '×': r'\ensuremath{\times}',
    'π': r'\ensuremath{\pi}',
    '−': r'\ensuremath{-}',
    '→': r'\ensuremath{\rightarrow}',
    '–': '--',
    '—': '---',
    '“': '``',
    '”': "''",
}
CITATION = re.compile(r'(?<!\[)\[(1[0-2]|[1-9])\](?!\])')
PLAIN_CAPTION = re.compile(r'^Figure(?:\s+[0-9A-D.]+)?\s*(?:[.:—-])?\s*')
TEX_REFERENCE = re.compile(r'\\ref\{(?:fig|tab|sec|app):[A-Za-z0-9:-]+\}')
TABLE_CAPTIONS = {
    ('02-methodology-and-evidence.md', 1): 'Final checkpoint against fixed-node Stockfish 13',
    ('05-systems-optimization.md', 1): 'Actor overlap: optimizer throughput and concurrent search',
    ('06-final-chess-recipe.md', 1): 'Final training run',
    ('04a-search.md', 1): 'Parallel search at 1,000 visits against 20,000-node Stockfish',
    ('07-final-run-results.md', 1): 'Distilled student strength across search budgets',
    ('appendix-c-supporting-comparisons.md', 1): 'CNN width: measured throughput versus arithmetic prediction',
    ('appendix-c-supporting-comparisons.md', 2): 'Depth and width: throughput ratios across batch sizes',
    ('appendix-c-supporting-comparisons.md', 3): 'Trainer throughput under actor overlap',
    ('appendix-c-supporting-comparisons.md', 4): 'Late-training checkpoint comparisons at two search budgets',
    ('appendix-d-reproducibility.md', 1): 'Chess input planes in tensor order (zero-based indices)',
}
MARKDOWN = MarkdownIt('commonmark').enable('table')


def escape_tex(value: str, *, citations: bool = True) -> str:
    """Escape prose while preserving only numbered public-source citations."""
    if TEX_REFERENCE.search(value):
        fragments: list[str] = []
        start = 0
        for reference in TEX_REFERENCE.finditer(value):
            fragments.append(escape_tex(value[start : reference.start()], citations=citations))
            fragments.append(reference.group())
            start = reference.end()
        fragments.append(escape_tex(value[start:], citations=citations))
        return ''.join(fragments)
    fragments = []
    start = 0
    matches = CITATION.finditer(value) if citations else ()
    for match in matches:
        fragments.append(''.join(SPECIAL_CHARACTERS.get(char, char) for char in value[start : match.start()]))
        fragments.append(r'\cite{ref' + match.group(1) + '}')
        start = match.end()
    fragments.append(''.join(SPECIAL_CHARACTERS.get(char, char) for char in value[start:]))
    return ''.join(fragments)


def inline_tex(tokens: list[Token], *, bibliography: bool = False) -> str:
    parts: list[str] = []
    links: list[str] = []
    for token in tokens:
        match token.type:
            case 'text':
                parts.append(escape_tex(token.content, citations=not bibliography))
            case 'code_inline':
                if token.content.startswith('deployment/'):
                    parts.append(r'\path{' + token.content + '}')
                else:
                    parts.append(r'\texttt{' + escape_tex(token.content, citations=False) + '}')
            case 'strong_open':
                parts.append(r'\textbf{')
            case 'em_open':
                parts.append(r'\emph{')
            case 'strong_close' | 'em_close':
                parts.append('}')
            case 'link_open':
                address = token.attrGet('href')
                if address in APPENDIX_FILES:
                    links.append(address)
                    parts.append(r'\hyperref[app:' + address[len('appendix-')].upper() + ']{')
                    continue
                if address in SOURCE_FILES:
                    links.append(address)
                    label = Path(address).stem
                    parts.append(r'\hyperref[sec:' + label + ']{')
                    continue
                if not bibliography:
                    raise ValueError('External links in report prose must be bibliography citations.')
                if address is None or urlparse(address).scheme not in {'https', 'http'}:
                    raise ValueError(f'Unsupported bibliography link: {address}')
                links.append(address)
                parts.append(r'\href{' + address.replace('%', r'\%') + '}{')
            case 'link_close':
                links.pop()
                parts.append('}')
            case 'softbreak' | 'hardbreak':
                parts.append(' ')
            case 'image':
                raise ValueError('Images must occupy their own Markdown paragraph.')
            case _:
                raise ValueError(f'Unsupported inline Markdown token: {token.type}')
    if links:
        raise ValueError('Unclosed Markdown hyperlink.')
    return ''.join(parts)


def abstract_tex() -> str:
    source = (REPORT_ROOT / '00-abstract.md').read_text(encoding='utf-8')
    content = source.split('## Abstract', maxsplit=1)[1].strip()
    return escape_tex(' '.join(content.split()))


def convert_figure(source: Path, build_directory: Path) -> str:
    resolved = source.resolve()
    if not resolved.is_relative_to(REPORT_ROOT.parent):
        raise ValueError(f'Figure escapes documentation root: {source}')
    output = build_directory / 'figures' / f'{resolved.stem}.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    drawing = svg2rlg(str(resolved))
    if drawing is None:
        raise ValueError(f'Cannot read figure: {source}')
    renderPDF.drawToFile(drawing, str(output))
    return output.as_posix()


def caption_tex(value: str, figure_name: str) -> str:
    body = PLAIN_CAPTION.sub('', value.strip('*').replace('**', ''), count=1).strip()
    return r'\caption{' + escape_tex(body) + r'}\label{fig:' + figure_name + '}'


def figure_tex(image: Token, caption: Token, source: Path, build_directory: Path, *, appendix: bool) -> str:
    address = image.attrGet('src')
    if address is None:
        raise ValueError('Figure has no source path.')
    figure_source = source.parent / address
    figure_path = convert_figure(figure_source, build_directory)
    width_points = figure_width_points(figure_source)
    caption_text = caption.content.replace('\n', ' ').strip()
    if appendix:
        environment, placement = 'figure', 'H'
    else:
        environment, placement = 'figure*', 't'
    return (
        f'\\begin{{{environment}}}[{placement}]\n'
        '\\centering\n'
        rf'\includegraphics[width={width_points:.3f}bp,height=0.70\textheight,keepaspectratio]{{'
        + figure_path
        + '}\n'
        + caption_tex(caption_text, Path(address).stem)
        + f'\n\\end{{{environment}}}\n'
    )


def figure_width_points(source: Path) -> float:
    svg = source.read_text(encoding='utf-8')
    root = element_tree.fromstring(svg)
    view_width = float(root.attrib['viewBox'].split()[2])
    font_sizes = [float(size) for size in re.findall(r'font-size:\s*([\d.]+)', svg)]
    font_sizes.extend(float(element.attrib['font-size']) for element in root.iter() if 'font-size' in element.attrib)
    for group in root.iter('{http://www.w3.org/2000/svg}g'):
        if group.attrib.get('id', '').startswith('text_'):
            for child in group:
                scale = re.search(r'scale\(([\d.]+)', child.attrib.get('transform', ''))
                if scale:
                    font_sizes.append(100 * float(scale.group(1)))
    if not font_sizes:
        raise ValueError(f'Cannot measure figure typography: {source}')
    text_width_points = (210 - 2 * 16) * 72 / 25.4
    return min(0.98 * text_width_points, 10 * view_width / max(font_sizes))


def parse_table(tokens: list[Token], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    row: list[str] = []
    index = start + 1
    while tokens[index].type != 'table_close':
        token = tokens[index]
        if token.type == 'tr_open':
            row = []
        elif token.type == 'tr_close':
            rows.append(row)
        elif token.type in {'th_open', 'td_open'}:
            content = tokens[index + 1]
            if content.type != 'inline':
                raise ValueError('Expected a Markdown table cell.')
            row.append(inline_tex(content.children or []))
        index += 1
    return rows, index + 1


def table_tex(rows: list[list[str]], *, source: Path, table_number: int, appendix: bool) -> str:
    columns = len(rows[0])
    if any(len(row) != columns for row in rows):
        raise ValueError('Inconsistent Markdown table width.')
    specification = '@{}' + 'l' * columns + '@{}'
    environment = 'table*' if source.name == '02-methodology-and-evidence.md' else 'table'
    placement = '[H]' if appendix else '[!t]'
    lines = [r'\begin{' + environment + '}' + placement]
    lines.append(r'\centering\normalsize' if appendix else r'\centering\small')
    caption = TABLE_CAPTIONS[(source.name, table_number)]
    lines.append(r'\caption{' + escape_tex(caption) + r'}\label{tab:' + source.stem + '-' + str(table_number) + '}')
    if appendix:
        lines.append(r'\setlength{\tabcolsep}{9pt}')
        lines.append(r'\renewcommand{\arraystretch}{1.1}')
    wrapped_input_table = source.name == 'appendix-d-reproducibility.md' and table_number == 1
    tabular_environment = 'tabularx' if wrapped_input_table else 'tabular'
    table_opening = (
        r'\begin{tabularx}{\textwidth}{@{}l X l@{}}'
        if wrapped_input_table
        else r'\begin{tabular}{' + specification + '}'
    )
    lines.extend([table_opening, r'\toprule'])
    for row_index, row in enumerate(rows):
        if row_index == 0 and not appendix:
            row = [
                r'\shortstack[l]{Estimated\\cycle time (s)}' if cell == 'Estimated cycle time (s)' else cell
                for cell in row
            ]
        lines.append(' & '.join(row) + r' \\')
        if row_index == 0:
            lines.append(r'\midrule')
    lines.extend([r'\bottomrule', r'\end{' + tabular_environment + '}'])
    lines.append(r'\end{' + environment + '}')
    return '\n'.join(lines) + '\n'


def section_tex(title: str, *, appendix: bool, level: int, appendix_letter: str, source: Path) -> str:
    if level == 1:
        if appendix:
            title = re.sub(r'^Appendix [A-D]\.\s*', '', title)
            return (
                r'\Needspace{6\baselineskip}\setcounter{figure}{0}\setcounter{table}{0}'
                + '\n'
                + r'\section{'
                + escape_tex(title)
                + r'}\label{app:'
                + appendix_letter
                + '}'
                + '\n'
            )
        heading = re.sub(r'^\d+(?:\.\d+)?\.\s*', '', title)
        if source.name.startswith(('04a-', '04b-', '04c-')):
            return r'\subsection{' + escape_tex(heading) + r'}\label{sec:' + source.stem + '}' + '\n'
        if source.name == '04-research-investigations.md':
            return r'\section{' + escape_tex(heading) + r'}\label{sec:research}\label{sec:' + source.stem + '}' + '\n'
        return r'\section{' + escape_tex(heading) + r'}\label{sec:' + source.stem + '}' + '\n'
    if level == 2:
        slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        label = r'\label{sec:' + source.stem + '-' + slug + '}'
        if source.name == 'appendix-d-reproducibility.md' and title == 'Result identity and provenance':
            return r'\Needspace{8\baselineskip}\subsection{' + escape_tex(title) + '}' + label + '\n'
        if source.name.startswith(('04a-', '04b-', '04c-')) and not appendix:
            return r'\subsubsection{' + escape_tex(title) + '}' + label + '\n'
        return r'\subsection{' + escape_tex(title) + '}' + label + '\n'
    return r'\paragraph{' + escape_tex(title) + '}' + '\n'


def markdown_tex(source: Path, build_directory: Path, *, appendix: bool = False) -> str:
    tokens = MARKDOWN.parse(source.read_text(encoding='utf-8'))
    lines: list[str] = []
    index = 0
    table_number = 0
    appendix_letter = source.name[len('appendix-')].upper() if appendix else ''
    while index < len(tokens):
        token = tokens[index]
        match token.type:
            case 'heading_open':
                content = tokens[index + 1]
                if content.type != 'inline':
                    raise ValueError('Expected heading text.')
                lines.append(
                    section_tex(
                        content.content,
                        appendix=appendix,
                        level=int(token.tag[1]),
                        appendix_letter=appendix_letter,
                        source=source,
                    )
                )
                index += 3
            case 'paragraph_open':
                content = tokens[index + 1]
                if content.type != 'inline':
                    raise ValueError('Expected paragraph text.')
                children = content.children or []
                if len(children) == 1 and children[0].type == 'image':
                    next_index = index + 3
                    if next_index >= len(tokens) or tokens[next_index].type != 'paragraph_open':
                        raise ValueError(f'Figure has no caption: {source}')
                    caption = tokens[next_index + 1]
                    lines.append(figure_tex(children[0], caption, source, build_directory, appendix=appendix))
                    index = next_index + 3
                else:
                    lines.append(inline_tex(children) + '\n\n')
                    index += 3
            case 'bullet_list_open':
                lines.append(r'\begin{itemize}' + '\n')
                index += 1
            case 'ordered_list_open':
                lines.append(r'\begin{enumerate}' + '\n')
                index += 1
            case 'bullet_list_close':
                lines.append(r'\end{itemize}' + '\n')
                index += 1
            case 'ordered_list_close':
                lines.append(r'\end{enumerate}' + '\n')
                index += 1
            case 'list_item_open':
                lines.append(r'\item ')
                index += 1
            case 'list_item_close':
                lines.append('\n')
                index += 1
            case 'table_open':
                rows, index = parse_table(tokens, index)
                table_number += 1
                lines.append(table_tex(rows, source=source, table_number=table_number, appendix=appendix))
            case 'paragraph_close':
                index += 1
            case 'fence' if token.info.strip() == 'math':
                lines.append('\\begin{equation}\n' + token.content.strip() + '\n\\end{equation}\n')
                index += 1
            case _:
                raise ValueError(f'Unsupported Markdown block token: {token.type} in {source}')
    return ''.join(lines)


def bibliography_tex() -> str:
    source = REPORT_ROOT / 'references-publication.md'
    tokens = MARKDOWN.parse(source.read_text(encoding='utf-8'))
    entries = [token for token in tokens if token.type == 'inline' and token.children and token.content]
    entries = entries[1:]
    if len(entries) != 12:
        raise ValueError(f'Expected 12 references, found {len(entries)}.')
    lines = [r'\begingroup\small', r'\begin{thebibliography}{12}']
    for number, entry in enumerate(entries, start=1):
        lines.append(r'\bibitem{ref' + str(number) + '} ' + inline_tex(entry.children or [], bibliography=True))
    lines.append(r'\end{thebibliography}')
    lines.append(r'\endgroup')
    return '\n'.join(lines) + '\n'


def find_tectonic() -> Path:
    executable = shutil.which('tectonic')
    if executable:
        return Path(executable)
    bundled = REPOSITORY_ROOT / 'tmp' / 'pdfs' / 'tectonic' / 'bin' / 'tectonic.exe'
    if bundled.exists():
        return bundled
    raise ValueError('Tectonic is required to build the report PDF. Install it and retry.')


def build_report(output: Path) -> None:
    build_directory = REPOSITORY_ROOT / 'tmp' / 'pdfs' / 'latex-build'
    build_directory.mkdir(parents=True, exist_ok=True)
    parts = [PREAMBLE, abstract_tex(), '\n', POST_ABSTRACT]
    for filename in SOURCE_FILES:
        parts.append(markdown_tex(REPORT_ROOT / filename, build_directory))
    parts = [''.join(parts)]
    parts.extend(
        [
            r'\FloatBarrier' + '\n',
            r'\balance' + '\n',
            bibliography_tex(),
            r'\clearpage\onecolumn\raggedbottom\widowpenalty=150\clubpenalty=150' + '\n',
            r'\appendix' + '\n',
            r'\renewcommand{\thefigure}{\Alph{section}.\arabic{figure}}' + '\n',
            r'\renewcommand{\thetable}{\Alph{section}.\arabic{table}}' + '\n',
        ]
    )
    for filename in APPENDIX_FILES:
        parts.append(markdown_tex(REPORT_ROOT / filename, build_directory, appendix=True))
    parts.append(r'\end{document}' + '\n')
    latex_source = build_directory / 'technical-report-review.tex'
    latex_source.write_text(''.join(parts), encoding='utf-8')
    command = [str(find_tectonic()), str(latex_source), '--outdir', str(build_directory), '--keep-logs']
    subprocess.run(command, check=True, cwd=REPOSITORY_ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(build_directory / 'technical-report-review.pdf', output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output', type=Path, default=REPOSITORY_ROOT / 'output' / 'pdf' / 'technical-report-review.pdf'
    )
    arguments = parser.parse_args()
    build_report(arguments.output.resolve())
    print(arguments.output.resolve())


if __name__ == '__main__':
    main()
