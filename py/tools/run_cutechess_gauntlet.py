from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GauntletConfiguration:
    cutechess: Path
    stockfish: Path
    model: Path
    openings: Path
    output_directory: Path
    games_per_opponent: int
    seconds_per_move: int
    stockfish_elos: tuple[int, ...]


def parse_arguments() -> GauntletConfiguration:
    parser = argparse.ArgumentParser(
        description="Run a paired-color Cute Chess calibration gauntlet."
    )
    parser.add_argument("--cutechess", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--openings", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--games-per-opponent", type=int, default=24)
    parser.add_argument("--seconds-per-move", type=int, default=10)
    parser.add_argument(
        "--stockfish-elo", type=int, action="append", default=[1600, 1800, 2000, 2200]
    )
    arguments = parser.parse_args()
    configuration = GauntletConfiguration(
        cutechess=arguments.cutechess,
        stockfish=arguments.stockfish,
        model=arguments.model,
        openings=arguments.openings,
        output_directory=arguments.output_directory,
        games_per_opponent=arguments.games_per_opponent,
        seconds_per_move=arguments.seconds_per_move,
        stockfish_elos=tuple(arguments.stockfish_elo),
    )
    validate_configuration(configuration)
    return configuration


def validate_configuration(configuration: GauntletConfiguration) -> None:
    for file_path in (
        configuration.cutechess,
        configuration.stockfish,
        configuration.model,
        configuration.openings,
    ):
        if not file_path.is_file():
            raise ValueError(f"Required file does not exist: {file_path}")
    if configuration.games_per_opponent < 2 or configuration.games_per_opponent % 2:
        raise ValueError("games_per_opponent must be a positive even number.")
    if not 1 <= configuration.seconds_per_move <= 30:
        raise ValueError("seconds_per_move must be in [1, 30].")


def command_for_opponent(
    configuration: GauntletConfiguration, stockfish_elo: int
) -> list[str]:
    pgn_path = configuration.output_directory / f"vs-stockfish-{stockfish_elo}.pgn"
    return [
        str(configuration.cutechess),
        "-engine",
        f"name=AlphaZeroCpp,cmd={sys.executable},dir={Path(__file__).parents[1]},proto=uci",
        "arg=-m",
        "arg=src.uci",
        "arg=--model",
        f"arg={configuration.model}",
        "-engine",
        f"name=Stockfish-{stockfish_elo},cmd={configuration.stockfish},proto=uci",
        "option.UCI_LimitStrength=true",
        f"option.UCI_Elo={stockfish_elo}",
        "-each",
        f"st={configuration.seconds_per_move}",
        "timemargin=2000",
        "-openings",
        f"file={configuration.openings}",
        "format=epd",
        "order=sequential",
        "policy=round",
        "-rounds",
        str(configuration.games_per_opponent),
        "-repeat",
        "-concurrency",
        "1",
        "-recover",
        "-pgnout",
        str(pgn_path),
        "fi",
    ]


def main() -> None:
    configuration = parse_arguments()
    configuration.output_directory.mkdir(parents=True, exist_ok=True)
    for stockfish_elo in configuration.stockfish_elos:
        subprocess.run(command_for_opponent(configuration, stockfish_elo), check=True)


if __name__ == "__main__":
    main()
