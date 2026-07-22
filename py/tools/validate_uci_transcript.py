from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import chess


@dataclass(frozen=True)
class ValidationRequest:
    transcript: Path
    moves_uci: tuple[str, ...]


def parse_arguments() -> ValidationRequest:
    parser = argparse.ArgumentParser(
        description="Validate a UCI startup and bestmove transcript."
    )
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--move", action="append", default=[])
    arguments = parser.parse_args()
    return ValidationRequest(arguments.transcript, tuple(arguments.move))


def validate_transcript(request: ValidationRequest) -> str:
    lines = request.transcript.read_text(encoding="utf-8").splitlines()
    if "uciok" not in lines:
        raise ValueError("Transcript does not contain uciok.")
    if "readyok" not in lines:
        raise ValueError("Transcript does not contain readyok.")

    bestmove_lines = [line for line in lines if line.startswith("bestmove ")]
    if len(bestmove_lines) != 1:
        raise ValueError(f"Expected one bestmove, found {len(bestmove_lines)}.")
    move_uci = bestmove_lines[0].removeprefix("bestmove ").split()[0]

    board = chess.Board()
    for history_move_uci in request.moves_uci:
        board.push_uci(history_move_uci)
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError as error:
        raise ValueError(f"Invalid bestmove UCI: {move_uci}") from error
    if move not in board.legal_moves:
        raise ValueError(f"Illegal bestmove {move_uci} for {board.fen()}")
    return move_uci


def main() -> None:
    request = parse_arguments()
    move_uci = validate_transcript(request)
    print(f"Validated legal bestmove: {move_uci}")


if __name__ == "__main__":
    main()
