from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from src.evaluation.katago_book import canonical_json_sha256, crawl_official_9x9_book, write_katago_book_export


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fetch a bounded, provenance-recorded KataGo 9x9 book export.')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--maximum-depth', required=True, type=int)
    parser.add_argument('--maximum-pages', required=True, type=int)
    parser.add_argument('--archive-html-root', type=Path)
    return parser.parse_args()


def archive_page_reader(html_root: Path) -> Callable[[str], bytes]:
    resolved_root = html_root.resolve()

    def read_page(url: str) -> bytes:
        path = urlparse(url).path
        prefix = '/book9x9tt/'
        if not path.startswith(prefix):
            raise ValueError(f'Unexpected KataGo book URL: {url}')
        resolved_path = (resolved_root / path.removeprefix(prefix)).resolve()
        if not resolved_path.is_relative_to(resolved_root):
            raise ValueError(f'KataGo book URL escapes the archive root: {url}')
        return resolved_path.read_bytes()

    return read_page


def main() -> None:
    arguments = parse_arguments()
    export = crawl_official_9x9_book(
        arguments.maximum_depth,
        arguments.maximum_pages,
        lambda pages, positions: print(f'Fetched {pages} pages; discovered {positions} positions', flush=True),
        None if arguments.archive_html_root is None else archive_page_reader(arguments.archive_html_root),
    )
    write_katago_book_export(arguments.output, export)
    print(f'Export: {arguments.output}')
    print(f'Pages: {len(export.pages)}')
    print(f'Positions: {len(export.positions)}')
    print(f'SHA-256: {canonical_json_sha256(arguments.output)}')


if __name__ == '__main__':
    main()
