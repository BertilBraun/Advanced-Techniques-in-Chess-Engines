"""Render the evidence-linked Markdown report as a two-column review PDF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path
from urllib.parse import quote

from markdown_it import MarkdownIt
from markdown_it.token import Token
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing, Group, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BalancedColumns,
    Flowable,
    HRFlowable,
    KeepTogether,
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)
from svglib.svglib import svg2rlg

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPOSITORY_ROOT / 'documentation' / 'report'
SOURCE_FILES = (
    '01-motivation-and-scope.md',
    '02-methodology-and-evidence.md',
    '03-system-and-methods.md',
    '04a-search.md',
    '04b-data-and-replay.md',
    '04c-networks-and-training.md',
    '05-systems-optimization.md',
    '05a-three-failures.md',
    '06-final-chess-recipe.md',
    '07-final-run-results.md',
    '08-limitations.md',
    '09-reproducibility.md',
    '10-conclusion.md',
)
PUBLIC_SOURCE_ROOT = 'https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/blob/master/'
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 48
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN
KIT_GREEN = colors.HexColor('#009682')
KIT_BLUE = colors.HexColor('#4664AA')
INK = colors.HexColor('#24282B')
MUTED = colors.HexColor('#5C686C')
PALE_GREEN = colors.HexColor('#ECF7F4')
PALE_BLUE = colors.HexColor('#F1F4FA')


@dataclass(frozen=True)
class ContentBlock:
    flowable: Flowable
    full_width: bool = False


class VectorFigure(Flowable):
    def __init__(self, source: Path, maximum_height: float = 345) -> None:
        super().__init__()
        drawing = svg2rlg(str(source))
        if drawing is None or drawing.width <= 0 or drawing.height <= 0:
            raise ValueError(f'Cannot render SVG figure: {source}')
        normalize_figure_fonts(drawing)
        self.drawing: Drawing = drawing
        self.maximum_height = maximum_height
        self.scale_factor = 1.0
        self.width = drawing.width
        self.height = drawing.height
        self.hAlign = 'CENTER'

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        self.scale_factor = min(
            available_width / self.drawing.width,
            self.maximum_height / self.drawing.height,
            1.0,
        )
        self.width = self.drawing.width * self.scale_factor
        self.height = self.drawing.height * self.scale_factor
        return self.width, self.height

    def draw(self) -> None:
        self.canv.saveState()
        self.canv.scale(self.scale_factor, self.scale_factor)
        renderPDF.draw(self.drawing, self.canv, 0, 0)
        self.canv.restoreState()


def normalize_figure_fonts(group: Drawing | Group) -> None:
    for item in group.contents:
        if isinstance(item, String):
            item.fontName = 'ReportSans-Bold' if 'bold' in item.fontName.lower() else 'ReportSans'
        elif isinstance(item, Group):
            normalize_figure_fonts(item)


def register_fonts() -> None:
    font_directory = Path('C:/Windows/Fonts')
    for name, filename in (
        ('ReportSerif', 'times.ttf'),
        ('ReportSerif-Bold', 'timesbd.ttf'),
        ('ReportSerif-Italic', 'timesi.ttf'),
        ('ReportSerif-BoldItalic', 'timesbi.ttf'),
        ('ReportSans', 'arial.ttf'),
        ('ReportSans-Bold', 'arialbd.ttf'),
    ):
        pdfmetrics.registerFont(TTFont(name, str(font_directory / filename)))
    pdfmetrics.registerFontFamily(
        'ReportSerif',
        normal='ReportSerif',
        bold='ReportSerif-Bold',
        italic='ReportSerif-Italic',
        boldItalic='ReportSerif-BoldItalic',
    )


def styles() -> dict[str, ParagraphStyle]:
    body = ParagraphStyle(
        'body',
        fontName='ReportSerif',
        fontSize=9.1,
        leading=11.4,
        textColor=INK,
        alignment=TA_JUSTIFY,
        spaceAfter=5.5,
        allowWidows=0,
        allowOrphans=0,
    )
    return {
        'body': body,
        'title': ParagraphStyle(
            'title',
            parent=body,
            fontName='ReportSans-Bold',
            fontSize=22,
            leading=25.5,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        'subtitle': ParagraphStyle(
            'subtitle',
            parent=body,
            fontName='ReportSans',
            fontSize=11,
            leading=14,
            textColor=KIT_GREEN,
            spaceAfter=13,
        ),
        'byline': ParagraphStyle(
            'byline',
            parent=body,
            fontName='ReportSans',
            fontSize=9,
            leading=12,
            textColor=MUTED,
            spaceAfter=16,
        ),
        'abstract_label': ParagraphStyle(
            'abstract_label',
            parent=body,
            fontName='ReportSans-Bold',
            fontSize=10,
            textColor=KIT_GREEN,
            spaceBefore=8,
            spaceAfter=5,
        ),
        'abstract': ParagraphStyle(
            'abstract',
            parent=body,
            fontSize=9.7,
            leading=12.4,
            spaceAfter=13,
        ),
        'h1': ParagraphStyle(
            'h1',
            parent=body,
            fontName='ReportSans-Bold',
            fontSize=13.0,
            leading=15.8,
            textColor=KIT_GREEN,
            spaceBefore=13,
            spaceAfter=7,
            keepWithNext=True,
        ),
        'h2': ParagraphStyle(
            'h2',
            parent=body,
            fontName='ReportSans-Bold',
            fontSize=10.1,
            leading=12.4,
            textColor=KIT_BLUE,
            spaceBefore=10,
            spaceAfter=4,
            keepWithNext=True,
        ),
        'h3': ParagraphStyle(
            'h3',
            parent=body,
            fontName='ReportSans-Bold',
            fontSize=9.3,
            leading=11.7,
            textColor=INK,
            spaceBefore=7,
            spaceAfter=3,
            keepWithNext=True,
        ),
        'list': ParagraphStyle(
            'list',
            parent=body,
            leftIndent=13,
            firstLineIndent=0,
            bulletIndent=2,
            spaceAfter=3.3,
        ),
        'quote': ParagraphStyle(
            'quote',
            parent=body,
            leftIndent=9,
            rightIndent=7,
            textColor=MUTED,
            borderColor=KIT_GREEN,
            borderWidth=1.1,
            borderPadding=7,
            spaceBefore=5,
            spaceAfter=8,
        ),
        'caption': ParagraphStyle(
            'caption',
            parent=body,
            fontName='ReportSerif-Italic',
            fontSize=8.2,
            leading=10.2,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=12,
        ),
        'table_cell': ParagraphStyle(
            'table_cell',
            parent=body,
            fontSize=8.2,
            leading=10.0,
            alignment=TA_LEFT,
            spaceAfter=0,
        ),
        'table_header': ParagraphStyle(
            'table_header',
            parent=body,
            fontName='ReportSans-Bold',
            fontSize=7.7,
            leading=9.2,
            textColor=colors.white,
            alignment=TA_LEFT,
            spaceAfter=0,
        ),
    }


def source_link(source: Path, href: str) -> str:
    if href.startswith(('https://', 'http://', 'mailto:')):
        return href
    target_name, separator, fragment = href.partition('#')
    if not target_name:
        return href
    target = (source.parent / target_name).resolve()
    try:
        repository_path = target.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return href
    result = PUBLIC_SOURCE_ROOT + quote(repository_path.as_posix())
    if separator:
        result += '#' + fragment
    return result


def inline_markup(tokens: list[Token], source: Path) -> str:
    rendered: list[str] = []
    for token in tokens:
        match token.type:
            case 'text':
                rendered.append(escape(token.content))
            case 'strong_open':
                rendered.append('<b>')
            case 'strong_close':
                rendered.append('</b>')
            case 'em_open':
                rendered.append('<i>')
            case 'em_close':
                rendered.append('</i>')
            case 'code_inline':
                rendered.append(f'<font face="Courier" size="8">{escape(token.content)}</font>')
            case 'link_open':
                href = source_link(source, token.attrGet('href') or '')
                rendered.append(f'<link href="{escape(href, quote=True)}" color="#4664AA">')
            case 'link_close':
                rendered.append('</link>')
            case 'softbreak':
                rendered.append(' ')
            case 'hardbreak':
                rendered.append('<br/>')
            case 'html_inline':
                rendered.append(escape(token.content))
            case 'image':
                rendered.append(escape(token.content))
            case _:
                if token.content:
                    rendered.append(escape(token.content))
    return ''.join(rendered)


def table_widths(column_count: int) -> list[float]:
    if column_count == 2:
        return [CONTENT_WIDTH * 0.26, CONTENT_WIDTH * 0.74]
    if column_count == 4:
        return [CONTENT_WIDTH * fraction for fraction in (0.16, 0.30, 0.33, 0.21)]
    return [CONTENT_WIDTH / column_count] * column_count


def make_table(rows: list[list[str]], report_styles: dict[str, ParagraphStyle]) -> LongTable:
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('Malformed Markdown table')
    cells = [
        [Paragraph(cell, report_styles['table_header' if row_index == 0 else 'table_cell']) for cell in row]
        for row_index, row in enumerate(rows)
    ]
    table = LongTable(cells, colWidths=table_widths(len(rows[0])), repeatRows=1, hAlign='CENTER')
    commands: list[tuple[object, ...]] = [
        ('BACKGROUND', (0, 0), (-1, 0), KIT_GREEN),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, 0), (-1, 0), 0.7, KIT_GREEN),
        ('LINEBELOW', (0, 1), (-1, -1), 0.25, colors.HexColor('#DCE6E3')),
    ]
    for row_index in range(2, len(rows), 2):
        commands.append(('BACKGROUND', (0, row_index), (-1, row_index), PALE_GREEN))
    table.setStyle(TableStyle(commands))
    return table


def parse_table(
    tokens: list[Token], start: int, source: Path, report_styles: dict[str, ParagraphStyle]
) -> tuple[ContentBlock, int]:
    rows: list[list[str]] = []
    current_row: list[str] = []
    position = start + 1
    while tokens[position].type != 'table_close':
        token = tokens[position]
        if token.type == 'tr_open':
            current_row = []
        elif token.type == 'inline':
            current_row.append(inline_markup(token.children or [], source))
        elif token.type == 'tr_close':
            rows.append(current_row)
        position += 1
    return ContentBlock(make_table(rows, report_styles), full_width=True), position + 1


def parse_markdown(
    source: Path, report_styles: dict[str, ParagraphStyle], content: str | None = None
) -> list[ContentBlock]:
    markdown = source.read_text(encoding='utf-8') if content is None else content
    tokens = MarkdownIt('commonmark').enable('table').parse(markdown)
    blocks: list[ContentBlock] = []
    list_depth = 0
    ordered_counters: list[int | None] = []
    in_quote = False
    position = 0
    while position < len(tokens):
        token = tokens[position]
        match token.type:
            case 'heading_open':
                heading = tokens[position + 1]
                level = token.tag
                style_name = level if level in {'h1', 'h2', 'h3'} else 'h3'
                blocks.append(
                    ContentBlock(Paragraph(inline_markup(heading.children or [], source), report_styles[style_name]))
                )
                position += 3
            case 'paragraph_open':
                inline = tokens[position + 1]
                children = inline.children or []
                images = [child for child in children if child.type == 'image']
                if len(images) == 1 and all(
                    child.type == 'image' or (child.type == 'text' and not child.content.strip()) for child in children
                ):
                    image_path = images[0].attrGet('src')
                    if image_path is None:
                        raise ValueError(f'Image has no source in {source}')
                    maximum_height = (
                        245
                        if image_path
                        in {
                            'figures/final-training-loss-and-rate.svg',
                            'figures/final-training-volume-and-throughput.svg',
                        }
                        else 345
                    )
                    figure = VectorFigure((source.parent / image_path).resolve(), maximum_height)
                    caption_text = escape(images[0].content)
                    following_position = position + 3
                    if (
                        following_position + 2 < len(tokens)
                        and tokens[following_position].type == 'paragraph_open'
                        and tokens[following_position + 1].type == 'inline'
                        and tokens[following_position + 1].content.lstrip('*').startswith('Figure ')
                        and tokens[following_position + 2].type == 'paragraph_close'
                    ):
                        caption_text = inline_markup(tokens[following_position + 1].children or [], source)
                        position = following_position
                    caption = Paragraph(caption_text, report_styles['caption'])
                    blocks.append(ContentBlock(KeepTogether([Spacer(1, 8), figure, caption]), full_width=True))
                else:
                    content = inline_markup(children, source)
                    if content.strip():
                        style_name = 'quote' if in_quote else 'list' if list_depth else 'body'
                        if content.lstrip().startswith('<b>Figure'):
                            style_name = 'caption'
                        bullet = None
                        if list_depth:
                            counter = ordered_counters[-1]
                            bullet = f'{counter}.' if counter is not None else '•'
                        blocks.append(ContentBlock(Paragraph(content, report_styles[style_name], bulletText=bullet)))
                position += 3
            case 'table_open':
                block, position = parse_table(tokens, position, source, report_styles)
                blocks.append(block)
            case 'bullet_list_open':
                list_depth += 1
                ordered_counters.append(None)
                position += 1
            case 'ordered_list_open':
                list_depth += 1
                ordered_counters.append(0)
                position += 1
            case 'list_item_open':
                if ordered_counters and ordered_counters[-1] is not None:
                    ordered_counters[-1] += 1
                position += 1
            case 'bullet_list_close' | 'ordered_list_close':
                list_depth -= 1
                ordered_counters.pop()
                position += 1
            case 'blockquote_open':
                in_quote = True
                position += 1
            case 'blockquote_close':
                in_quote = False
                position += 1
            case 'fence' | 'code_block':
                blocks.append(
                    ContentBlock(
                        Paragraph(
                            f'<font face="Courier" size="8">{escape(token.content).replace(chr(10), "<br/>")}</font>',
                            report_styles['quote'],
                        )
                    )
                )
                position += 1
            case _:
                position += 1
    return blocks


def append_column_content(story: list[Flowable], pending: list[Flowable]) -> None:
    if pending:
        story.append(
            BalancedColumns(
                pending[:],
                nCols=2,
                needed=58,
                innerPadding=17,
                leftPadding=0,
                rightPadding=0,
                topPadding=0,
                bottomPadding=0,
            )
        )
        pending.clear()


def draw_page(canvas: Canvas, document: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setTitle('Engineering Efficient Self-Play Chess')
    canvas.setAuthor('Bertil Braun')
    canvas.setSubject('Compute-constrained AlphaZero-style chess technical report')
    page = document.page
    canvas.setStrokeColor(KIT_GREEN)
    canvas.setLineWidth(0.7)
    canvas.line(MARGIN, PAGE_HEIGHT - 34, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 34)
    canvas.setFont('ReportSans', 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 33, 'Engineering Efficient Self-Play Chess · research report')
    canvas.setFillColor(KIT_BLUE)
    canvas.drawRightString(PAGE_WIDTH - MARGIN, 33, str(page))
    canvas.restoreState()


def abstract_text() -> str:
    source = REPORT_ROOT / '00-abstract.md'
    lines = source.read_text(encoding='utf-8').splitlines()
    abstract_start = lines.index('## Abstract') + 1
    return ' '.join(line.strip() for line in lines[abstract_start:] if line.strip())


def build_report(output: Path) -> None:
    register_fonts()
    report_styles = styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=46,
        bottomMargin=48,
        title='Engineering Efficient Self-Play Chess',
        author='Bertil Braun',
    )
    story: list[Flowable] = [
        Paragraph('Engineering Efficient Self-Play Chess', report_styles['title']),
        Paragraph(
            'Search, replay, architecture, and throughput under limited training compute',
            report_styles['subtitle'],
        ),
        Paragraph('Bertil Braun · September 2026 · Technical report', report_styles['byline']),
        HRFlowable(width='100%', thickness=2, color=KIT_GREEN, spaceAfter=10),
        Paragraph('Abstract', report_styles['abstract_label']),
        Paragraph(escape(abstract_text()), report_styles['abstract']),
        HRFlowable(width='100%', thickness=0.5, color=KIT_BLUE, spaceAfter=11),
    ]
    pending: list[Flowable] = []
    for filename in SOURCE_FILES:
        for block in parse_markdown(REPORT_ROOT / filename, report_styles):
            if block.full_width:
                append_column_content(story, pending)
                story.append(block.flowable)
            else:
                pending.append(block.flowable)
    reference_styles = report_styles | {
        'list': ParagraphStyle(
            'reference_list',
            parent=report_styles['list'],
            fontSize=7.9,
            leading=9.1,
            spaceAfter=1.3,
        )
    }
    for block in parse_markdown(REPORT_ROOT / 'references-publication.md', reference_styles):
        pending.append(block.flowable)
    append_column_content(story, pending)
    document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        type=Path,
        default=REPOSITORY_ROOT / 'output' / 'pdf' / 'technical-report-review.pdf',
    )
    arguments = parser.parse_args()
    build_report(arguments.output.resolve())
    print(arguments.output.resolve())


if __name__ == '__main__':
    main()
