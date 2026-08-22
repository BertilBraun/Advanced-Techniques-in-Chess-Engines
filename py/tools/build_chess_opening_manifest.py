from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from src.evaluation.configuration import (
    ChessBookOpeningSource,
    OpeningSuiteConfiguration,
    StockfishEngineConfiguration,
)
from src.evaluation.openings import build_chess_book_opening_suite
from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.stockfish import StockfishClient
from src.games.chess.training import ChessImplementation
from src.util.hashing import file_sha256


def _source_revision() -> str:
    return subprocess.run(
        ('git', 'rev-parse', 'HEAD'),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_manifest(
    experiment_path: Path,
    selection_path: Path,
    stockfish_executable: Path,
    output_path: Path,
) -> None:
    loaded = load_experiment_configuration(experiment_path)
    if not isinstance(loaded, ChessExperimentConfiguration):
        raise ValueError('Chess opening manifest builder requires a chess experiment.')
    configured_engine = loaded.evaluation.engine
    if not isinstance(configured_engine, StockfishEngineConfiguration):
        raise ValueError('Chess opening manifest builder requires a Stockfish evaluation engine.')
    selection_rows = tuple(
        line for line in selection_path.read_text(encoding='utf-8').splitlines() if line and not line.startswith('#')
    )
    configuration = OpeningSuiteConfiguration(
        path=str(output_path),
        opening_count=len(selection_rows),
        source=ChessBookOpeningSource(
            kind='chess_book',
            selection_path=str(selection_path),
            selection_sha256=file_sha256(selection_path),
        ),
    )
    engine_configuration = StockfishEngineConfiguration(
        kind='stockfish',
        executable_path=str(stockfish_executable.resolve()),
        label_nodes=configured_engine.label_nodes,
        match_nodes=configured_engine.match_nodes,
        threads=configured_engine.threads,
        hash_mib=configured_engine.hash_mib,
        multi_pv=configured_engine.multi_pv,
        policy_softmax_temperature=configured_engine.policy_softmax_temperature,
    )
    game = ChessImplementation(loaded)
    with StockfishClient(engine_configuration, game.state, stockfish_executable.resolve()) as engine:
        build_chess_book_opening_suite(
            output_path,
            selection_path,
            configuration,
            game.state,
            engine,
            _source_revision(),
        )


def parse_arguments() -> tuple[Path, Path, Path, Path]:
    parser = argparse.ArgumentParser(description='Build an immutable chess opening manifest from a reviewed TSV.')
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--stockfish-executable', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    namespace = parser.parse_args()
    inputs = (namespace.experiment, namespace.selection, namespace.stockfish_executable)
    if not all(path.is_file() for path in inputs):
        raise ValueError('Experiment, selection, and Stockfish executable must be files.')
    if namespace.output.exists():
        raise ValueError(f'Opening manifest output already exists: {namespace.output}')
    return namespace.experiment, namespace.selection, namespace.stockfish_executable, namespace.output


def main() -> None:
    build_manifest(*parse_arguments())


if __name__ == '__main__':
    main()
