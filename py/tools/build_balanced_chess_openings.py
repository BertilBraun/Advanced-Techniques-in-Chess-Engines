from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Literal

import chess
import chess.engine
import chess.pgn
from pydantic import Field

from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


@dataclass
class _PositionEvidence:
    frequency: int = 0
    lines: Counter[tuple[tuple[str, ...], str]] = field(default_factory=Counter)
    eco_codes: Counter[str] = field(default_factory=Counter)
    opening_names: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class _Candidate:
    moves_uci: tuple[str, ...]
    final_fen: str
    frequency: int
    eco_code: str
    opening_name: str


@dataclass(frozen=True)
class _OpeningRecord:
    moves_uci: tuple[str, ...]
    final_fen: str
    eco_code: str
    opening_name: str


class _OpeningVisitor(chess.pgn.BaseVisitor[_OpeningRecord | None]):
    def __init__(self, minimum_player_rating: int, opening_plies: int) -> None:
        self._minimum_player_rating = minimum_player_rating
        self._opening_plies = opening_plies
        self._headers = chess.pgn.Headers({})
        self._moves_uci: list[str] = []
        self._final_board: chess.Board | None = None

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        self._headers[tagname] = tagvalue

    def end_headers(self) -> chess.pgn.SkipType | None:
        ratings = (self._headers.get('WhiteElo', ''), self._headers.get('BlackElo', ''))
        if (
            self._headers.get('Variant', 'Standard') != 'Standard'
            or self._headers.get('SetUp', '0') != '0'
            or not all(rating.isdigit() for rating in ratings)
            or min(int(rating) for rating in ratings) < self._minimum_player_rating
        ):
            return chess.pgn.SKIP
        return None

    def begin_variation(self) -> chess.pgn.SkipType:
        return chess.pgn.SKIP

    def begin_parse_san(self, board: chess.Board, san: str) -> chess.pgn.SkipType | None:
        del board, san
        return chess.pgn.SKIP if len(self._moves_uci) >= self._opening_plies else None

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        self._moves_uci.append(move.uci())
        if len(self._moves_uci) == self._opening_plies:
            final_board = board.copy(stack=False)
            final_board.push(move)
            self._final_board = final_board

    def result(self) -> _OpeningRecord | None:
        board = self._final_board
        if board is None or board.is_game_over() or board.is_check() or not _has_equal_material(board):
            return None
        return _OpeningRecord(
            moves_uci=tuple(self._moves_uci),
            final_fen=board.fen(),
            eco_code=self._headers.get('ECO', 'A00'),
            opening_name=self._headers.get('Opening', 'Unclassified'),
        )


class SelectedOpening(FrozenModel):
    opening_id: str = Field(min_length=1)
    moves_uci: tuple[str, ...] = Field(min_length=2)
    final_fen: str = Field(min_length=1)
    source_game_count: int = Field(gt=0)
    eco_code: str = Field(min_length=3, max_length=3)
    opening_name: str = Field(min_length=1)
    stockfish_centipawns_white: int
    stockfish_wdl_white: tuple[int, int, int]


class OpeningSelectionReport(FrozenModel):
    schema_version: Literal[1] = 1
    source_url: str = Field(min_length=1)
    source_archive_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_pgn_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_games_read: int = Field(gt=0)
    eligible_source_games: int = Field(gt=0)
    minimum_player_rating: int = Field(gt=0)
    opening_plies: int = Field(gt=0)
    maximum_absolute_centipawns: int = Field(gt=0)
    maximum_openings_per_eco: int = Field(gt=0)
    stockfish_identity: str = Field(min_length=1)
    stockfish_executable_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    stockfish_nodes: int = Field(gt=0)
    stockfish_threads: int = Field(gt=0)
    stockfish_hash_mib: int = Field(gt=0)
    candidates_evaluated: int = Field(gt=0)
    selected_openings: tuple[SelectedOpening, ...] = Field(min_length=1)


@dataclass(frozen=True)
class Arguments:
    source_pgn: Path
    source_archive: Path
    source_url: str
    stockfish_executable: Path
    stockfish_nodes: int
    stockfish_threads: int
    stockfish_hash_mib: int
    minimum_player_rating: int
    opening_plies: int
    opening_count: int
    maximum_absolute_centipawns: int
    maximum_openings_per_eco: int
    output_selection: Path
    output_report: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _has_equal_material(board: chess.Board) -> bool:
    return all(
        len(board.pieces(piece_type, chess.WHITE)) == len(board.pieces(piece_type, chess.BLACK))
        for piece_type in range(chess.PAWN, chess.KING + 1)
    )


def _extract_candidates(
    source_pgn: Path,
    minimum_player_rating: int,
    opening_plies: int,
) -> tuple[tuple[_Candidate, ...], int, int]:
    evidence_by_position: dict[str, _PositionEvidence] = {}
    games_read = 0
    eligible_games = 0
    with source_pgn.open(encoding='utf-8', errors='replace') as pgn_file:
        while True:
            position_before = pgn_file.tell()
            record = chess.pgn.read_game(
                pgn_file,
                Visitor=lambda: _OpeningVisitor(minimum_player_rating, opening_plies),
            )
            if record is None and pgn_file.tell() == position_before:
                break
            games_read += 1
            if record is None:
                continue
            eligible_games += 1
            board = chess.Board(record.final_fen)
            canonical_position = board.epd()
            evidence = evidence_by_position.setdefault(canonical_position, _PositionEvidence())
            evidence.frequency += 1
            evidence.lines[(record.moves_uci, record.final_fen)] += 1
            evidence.eco_codes[record.eco_code] += 1
            evidence.opening_names[record.opening_name] += 1
    candidates = tuple(
        sorted(
            (
                _Candidate(
                    moves_uci=evidence.lines.most_common(1)[0][0][0],
                    final_fen=evidence.lines.most_common(1)[0][0][1],
                    frequency=evidence.frequency,
                    eco_code=evidence.eco_codes.most_common(1)[0][0],
                    opening_name=evidence.opening_names.most_common(1)[0][0],
                )
                for evidence in evidence_by_position.values()
            ),
            key=lambda candidate: (-candidate.frequency, candidate.moves_uci),
        )
    )
    return candidates, games_read, eligible_games


def _evaluate_candidate(
    engine: chess.engine.SimpleEngine,
    candidate: _Candidate,
    nodes: int,
) -> tuple[int, tuple[int, int, int]]:
    engine.configure({'Clear Hash': None})
    board = chess.Board(candidate.final_fen)
    information = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        info=chess.engine.INFO_SCORE,
    )
    score = information.get('score')
    wdl = information.get('wdl')
    if not isinstance(score, chess.engine.PovScore) or not isinstance(wdl, chess.engine.PovWdl):
        raise ValueError('Stockfish opening analysis omitted score or WDL information.')
    centipawns = score.white().score(mate_score=100_000)
    if centipawns is None:
        raise ValueError('Stockfish opening analysis returned an unusable score.')
    white_wdl = wdl.white()
    return centipawns, (white_wdl.wins, white_wdl.draws, white_wdl.losses)


def _select_openings(
    candidates: tuple[_Candidate, ...],
    arguments: Arguments,
) -> tuple[tuple[SelectedOpening, ...], str, int]:
    engine = chess.engine.SimpleEngine.popen_uci(str(arguments.stockfish_executable))
    engine.configure(
        {
            'Threads': arguments.stockfish_threads,
            'Hash': arguments.stockfish_hash_mib,
            'UCI_ShowWDL': True,
        }
    )
    selected: list[SelectedOpening] = []
    eco_counts: Counter[str] = Counter()
    evaluated = 0
    try:
        identity = str(engine.id.get('name', 'Stockfish'))
        for candidate in candidates:
            if eco_counts[candidate.eco_code] >= arguments.maximum_openings_per_eco:
                continue
            centipawns, wdl = _evaluate_candidate(engine, candidate, arguments.stockfish_nodes)
            evaluated += 1
            if abs(centipawns) > arguments.maximum_absolute_centipawns:
                continue
            opening_id = f'elite-2025-11-{len(selected) + 1:03d}'
            selected.append(
                SelectedOpening(
                    opening_id=opening_id,
                    moves_uci=candidate.moves_uci,
                    final_fen=candidate.final_fen,
                    source_game_count=candidate.frequency,
                    eco_code=candidate.eco_code,
                    opening_name=candidate.opening_name,
                    stockfish_centipawns_white=centipawns,
                    stockfish_wdl_white=wdl,
                )
            )
            eco_counts[candidate.eco_code] += 1
            if len(selected) == arguments.opening_count:
                return tuple(selected), identity, evaluated
    finally:
        engine.quit()
    raise ValueError(
        f'Only {len(selected)} openings passed selection after evaluating {evaluated} candidates; '
        'relax the declared constraints.'
    )


def _selection_text(arguments: Arguments, report: OpeningSelectionReport) -> str:
    header = (
        '# Balanced high-frequency opening positions from the Lichess Elite November 2025 corpus.\n'
        f'# Source URL: {arguments.source_url}\n'
        f'# Source archive SHA-256: {report.source_archive_sha256}\n'
        f'# Source PGN SHA-256: {report.source_pgn_sha256}\n'
        f'# Selection: both ratings >= {arguments.minimum_player_rating}, {arguments.opening_plies} plies, equal '
        f'material, not in check, unique canonical positions, |SF eval| <= '
        f'{arguments.maximum_absolute_centipawns} cp, max {arguments.maximum_openings_per_eco} per ECO.\n'
        f'# Stockfish: {report.stockfish_identity}; nodes={arguments.stockfish_nodes}; '
        f'threads={arguments.stockfish_threads}; hash_mib={arguments.stockfish_hash_mib}; '
        f'sha256={report.stockfish_executable_sha256}\n'
        '# Columns: opening_id, UCI move sequence, expected final FEN\n'
    )
    rows = ''.join(
        f'{opening.opening_id}\t{" ".join(opening.moves_uci)}\t{opening.final_fen}\n'
        for opening in report.selected_openings
    )
    return header + rows


def build_selection(arguments: Arguments) -> OpeningSelectionReport:
    candidates, games_read, eligible_games = _extract_candidates(
        arguments.source_pgn,
        arguments.minimum_player_rating,
        arguments.opening_plies,
    )
    selected, stockfish_identity, candidates_evaluated = _select_openings(candidates, arguments)
    report = OpeningSelectionReport(
        source_url=arguments.source_url,
        source_archive_sha256=_sha256(arguments.source_archive),
        source_pgn_sha256=_sha256(arguments.source_pgn),
        source_games_read=games_read,
        eligible_source_games=eligible_games,
        minimum_player_rating=arguments.minimum_player_rating,
        opening_plies=arguments.opening_plies,
        maximum_absolute_centipawns=arguments.maximum_absolute_centipawns,
        maximum_openings_per_eco=arguments.maximum_openings_per_eco,
        stockfish_identity=stockfish_identity,
        stockfish_executable_sha256=_sha256(arguments.stockfish_executable),
        stockfish_nodes=arguments.stockfish_nodes,
        stockfish_threads=arguments.stockfish_threads,
        stockfish_hash_mib=arguments.stockfish_hash_mib,
        candidates_evaluated=candidates_evaluated,
        selected_openings=selected,
    )
    write_text_atomically(arguments.output_selection, _selection_text(arguments, report))
    write_text_atomically(arguments.output_report, report.model_dump_json(indent=2) + '\n')
    return report


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Build a balanced opening suite from a frozen elite-game PGN.')
    parser.add_argument('--source-pgn', required=True, type=Path)
    parser.add_argument('--source-archive', required=True, type=Path)
    parser.add_argument('--source-url', required=True)
    parser.add_argument('--stockfish-executable', required=True, type=Path)
    parser.add_argument('--stockfish-nodes', default=250_000, type=int)
    parser.add_argument('--stockfish-threads', default=1, type=int)
    parser.add_argument('--stockfish-hash-mib', default=128, type=int)
    parser.add_argument('--minimum-player-rating', default=2_500, type=int)
    parser.add_argument('--opening-plies', default=8, type=int)
    parser.add_argument('--opening-count', default=200, type=int)
    parser.add_argument('--maximum-absolute-centipawns', default=50, type=int)
    parser.add_argument('--maximum-openings-per-eco', default=4, type=int)
    parser.add_argument('--output-selection', required=True, type=Path)
    parser.add_argument('--output-report', required=True, type=Path)
    namespace = parser.parse_args()
    arguments = Arguments(
        source_pgn=namespace.source_pgn,
        source_archive=namespace.source_archive,
        source_url=namespace.source_url,
        stockfish_executable=namespace.stockfish_executable,
        stockfish_nodes=namespace.stockfish_nodes,
        stockfish_threads=namespace.stockfish_threads,
        stockfish_hash_mib=namespace.stockfish_hash_mib,
        minimum_player_rating=namespace.minimum_player_rating,
        opening_plies=namespace.opening_plies,
        opening_count=namespace.opening_count,
        maximum_absolute_centipawns=namespace.maximum_absolute_centipawns,
        maximum_openings_per_eco=namespace.maximum_openings_per_eco,
        output_selection=namespace.output_selection,
        output_report=namespace.output_report,
    )
    if (
        not arguments.source_pgn.is_file()
        or not arguments.source_archive.is_file()
        or not arguments.stockfish_executable.is_file()
    ):
        raise ValueError('Source archive, source PGN, and Stockfish executable must exist.')
    positive_values = (
        arguments.stockfish_nodes,
        arguments.stockfish_threads,
        arguments.stockfish_hash_mib,
        arguments.minimum_player_rating,
        arguments.opening_plies,
        arguments.opening_count,
        arguments.maximum_absolute_centipawns,
        arguments.maximum_openings_per_eco,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError('Numeric selection parameters must be positive.')
    if arguments.opening_plies % 2:
        raise ValueError('Opening plies must be even so every opening ends with White to move.')
    if arguments.output_selection.exists() or arguments.output_report.exists():
        raise ValueError('Opening selection outputs must not already exist.')
    return arguments


def main() -> None:
    print(json.dumps(json.loads(build_selection(parse_arguments()).model_dump_json()), indent=2))


if __name__ == '__main__':
    main()
