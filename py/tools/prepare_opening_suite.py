from __future__ import annotations

import argparse
import hashlib
import io
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

import chess.pgn


STOCKFISH_BOOKS_COMMIT = '65815ccdbc7727cd4f6aee252ba8f67fb740e92f'
EIGHT_MOVES_V3_SHA256 = '7e1e9dd118b4bb97d8a8b5b8a790c86e21f8509d59a27d2883767d94477be02e'
ARCHIVE_MEMBER = '8moves_v3.pgn'


@dataclass(frozen=True)
class CommandLineArguments:
    archive_path: Path
    output_path: Path
    opening_count: int
    random_seed: int


def parse_arguments() -> CommandLineArguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('archive_path', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('--opening-count', type=int, required=True)
    parser.add_argument('--random-seed', type=int, required=True)
    namespace = parser.parse_args()
    return CommandLineArguments(
        archive_path=namespace.archive_path,
        output_path=namespace.output_path,
        opening_count=namespace.opening_count,
        random_seed=namespace.random_seed,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_opening_fens(archive_path: Path, expected_sha256: str) -> tuple[str, ...]:
    actual_sha256 = file_sha256(archive_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f'Opening archive SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}.')

    fens: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != [ARCHIVE_MEMBER]:
            raise ValueError(f'Expected only {ARCHIVE_MEMBER!r} in opening archive.')
        pgn_text = archive.read(ARCHIVE_MEMBER).decode('utf-8')

    pgn_stream = io.StringIO(pgn_text)
    while game := chess.pgn.read_game(pgn_stream):
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
        fens.append(board.fen())
    return tuple(fens)


def select_openings(
    fens: tuple[str, ...],
    opening_count: int,
    random_seed: int,
) -> tuple[tuple[int, str], ...]:
    if opening_count < 1:
        raise ValueError('opening_count must be positive.')
    if opening_count > len(fens):
        raise ValueError(f'Requested {opening_count} openings from a book containing {len(fens)}.')
    random_number_generator = random.Random(random_seed)
    selected_indices = sorted(random_number_generator.sample(range(len(fens)), opening_count))
    return tuple((index, fens[index]) for index in selected_indices)


def write_opening_suite(path: Path, openings: tuple[tuple[int, str], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f'.{path.name}.tmp')
    lines = [
        '# Deterministic subset of official-stockfish/books 8moves_v3.pgn.',
        f'# Source commit: {STOCKFISH_BOOKS_COMMIT}',
        f'# Source archive SHA-256: {EIGHT_MOVES_V3_SHA256}',
    ]
    lines.extend(f'8moves-v3-{source_index:05d}\t{fen}' for source_index, fen in openings)
    temporary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temporary_path.replace(path)


def prepare_opening_suite(
    archive_path: Path,
    output_path: Path,
    opening_count: int,
    random_seed: int,
    expected_sha256: str = EIGHT_MOVES_V3_SHA256,
) -> None:
    fens = read_opening_fens(archive_path, expected_sha256)
    openings = select_openings(fens, opening_count, random_seed)
    write_opening_suite(output_path, openings)


def main() -> None:
    arguments = parse_arguments()
    prepare_opening_suite(
        archive_path=arguments.archive_path,
        output_path=arguments.output_path,
        opening_count=arguments.opening_count,
        random_seed=arguments.random_seed,
    )


if __name__ == '__main__':
    main()
