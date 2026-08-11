from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.katago_book import canonical_json_sha256, crawl_official_9x9_book, write_katago_book_export


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fetch a bounded, provenance-recorded KataGo 9x9 book export.')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--maximum-depth', required=True, type=int)
    parser.add_argument('--maximum-pages', required=True, type=int)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    export = crawl_official_9x9_book(
        arguments.maximum_depth,
        arguments.maximum_pages,
        lambda pages, positions: print(f'Fetched {pages} pages; discovered {positions} positions', flush=True),
    )
    write_katago_book_export(arguments.output, export)
    print(f'Export: {arguments.output}')
    print(f'Pages: {len(export.pages)}')
    print(f'Positions: {len(export.positions)}')
    print(f'SHA-256: {canonical_json_sha256(arguments.output)}')


if __name__ == '__main__':
    main()
