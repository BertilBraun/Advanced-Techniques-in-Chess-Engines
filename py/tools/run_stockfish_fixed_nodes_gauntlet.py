from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
import time
from typing import Literal

from pydantic import Field

from src.evaluation.configuration import (
    StockfishEngineConfiguration,
    StockfishFixedNodesEvaluationDefinition,
)
from src.evaluation.contracts import (
    MatchEvaluationJob,
    MatchEvaluationResult,
    OPENING_SUITE_MANIFEST_ADAPTER,
    StockfishFixedNodesOpponent,
)
from src.evaluation.match import run_match
from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.stockfish import StockfishClient, StockfishFixedNodesMatchEngine
from src.games.chess.training import ChessImplementation
from src.training.checkpoint import CheckpointReference
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


@dataclass(frozen=True)
class Arguments:
    experiment: Path
    run_directory: Path
    checkpoint_generation: int
    opening_manifest: Path
    stockfish_executable: Path
    match_nodes: int
    opening_pairs: int
    device_id: int
    output: Path


class StockfishFixedNodesGauntletResult(FrozenModel):
    schema_version: Literal[1] = 1
    source_revision: str = Field(min_length=40, max_length=40)
    tool_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    experiment_path: Path
    run_directory: Path
    opening_manifest_path: Path
    opening_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    stockfish_executable_path: Path
    stockfish_executable_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    stockfish_identity: str = Field(min_length=1)
    stockfish_match_nodes: int = Field(gt=0)
    match: MatchEvaluationResult
    duration_seconds: float = Field(ge=0.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision() -> str:
    return subprocess.run(
        ('git', 'rev-parse', 'HEAD'),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixed_nodes_definition(
    configuration: ChessExperimentConfiguration,
    opening_pairs: int,
    match_nodes: int,
) -> StockfishFixedNodesEvaluationDefinition:
    for definition in configuration.evaluation.definitions:
        if isinstance(definition, StockfishFixedNodesEvaluationDefinition):
            return definition.model_copy(
                update={
                    'definition_id': f'stockfish-13-fixed-nodes-{match_nodes}',
                    'opening_pair_count': opening_pairs,
                }
            )
    raise ValueError('Chess gauntlet requires a configured fixed-node Stockfish definition.')


def _stockfish_configuration(
    configuration: ChessExperimentConfiguration,
    executable: Path,
    match_nodes: int,
) -> StockfishEngineConfiguration:
    engine = configuration.evaluation.engine
    if not isinstance(engine, StockfishEngineConfiguration):
        raise ValueError('Chess gauntlet requires Stockfish engine configuration.')
    return StockfishEngineConfiguration(
        kind='stockfish',
        executable_path=str(executable.resolve()),
        label_nodes=engine.label_nodes,
        match_nodes=match_nodes,
        threads=engine.threads,
        hash_mib=engine.hash_mib,
        multi_pv=engine.multi_pv,
        policy_softmax_temperature=engine.policy_softmax_temperature,
    )


def run_gauntlet(arguments: Arguments) -> StockfishFixedNodesGauntletResult:
    started_at = time.monotonic()
    loaded = load_experiment_configuration(arguments.experiment)
    if not isinstance(loaded, ChessExperimentConfiguration):
        raise ValueError('Stockfish fixed-node gauntlet requires a chess experiment.')
    checkpoint = CheckpointReference.load_for_inference(
        arguments.run_directory,
        arguments.checkpoint_generation,
    )
    openings = OPENING_SUITE_MANIFEST_ADAPTER.validate_json(arguments.opening_manifest.read_text(encoding='utf-8'))
    if openings.game != 'chess':
        raise ValueError('Stockfish fixed-node gauntlet requires chess openings.')
    definition = _fixed_nodes_definition(loaded, arguments.opening_pairs, arguments.match_nodes)
    job = MatchEvaluationJob(
        kind='match',
        job_id=f'stockfish-13-fixed-nodes-{arguments.match_nodes}-g{checkpoint.generation}',
        definition=definition,
        boundary_seconds=1,
        candidate=checkpoint,
        opponent=StockfishFixedNodesOpponent(kind='stockfish_fixed_nodes'),
        device_id=arguments.device_id,
        deadline_seconds=7 * 24 * 60 * 60,
        random_seed=loaded.training.random_seed + arguments.match_nodes,
        result_path=arguments.output.resolve(),
    )
    game = ChessImplementation(loaded)
    engine_configuration = _stockfish_configuration(loaded, arguments.stockfish_executable, arguments.match_nodes)
    client = StockfishClient(engine_configuration, game.state, arguments.stockfish_executable.resolve())
    stockfish_identity = client.engine_identity
    external_engine = StockfishFixedNodesMatchEngine(client)
    try:
        match = run_match(
            job,
            game,
            openings,
            loaded.evaluation.bootstrap_samples,
            external_engine,
            loaded.training.topology.trainer.device_type,
        )
    finally:
        external_engine.close()
    result = StockfishFixedNodesGauntletResult(
        source_revision=_source_revision(),
        tool_sha256=_sha256(Path(__file__)),
        experiment_path=arguments.experiment.resolve(),
        run_directory=arguments.run_directory.resolve(),
        opening_manifest_path=arguments.opening_manifest.resolve(),
        opening_manifest_sha256=_sha256(arguments.opening_manifest),
        stockfish_executable_path=arguments.stockfish_executable.resolve(),
        stockfish_executable_sha256=_sha256(arguments.stockfish_executable),
        stockfish_identity=stockfish_identity,
        stockfish_match_nodes=arguments.match_nodes,
        match=match,
        duration_seconds=time.monotonic() - started_at,
    )
    write_text_atomically(arguments.output, result.model_dump_json(indent=2) + '\n')
    return result


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Run a paired model-versus-Stockfish fixed-node gauntlet.')
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--run-directory', required=True, type=Path)
    parser.add_argument('--checkpoint-generation', required=True, type=int)
    parser.add_argument('--opening-manifest', required=True, type=Path)
    parser.add_argument('--stockfish-executable', required=True, type=Path)
    parser.add_argument('--match-nodes', required=True, type=int)
    parser.add_argument('--opening-pairs', default=50, type=int)
    parser.add_argument('--device-id', default=7, type=int)
    parser.add_argument('--output', required=True, type=Path)
    namespace = parser.parse_args()
    arguments = Arguments(
        experiment=namespace.experiment,
        run_directory=namespace.run_directory,
        checkpoint_generation=namespace.checkpoint_generation,
        opening_manifest=namespace.opening_manifest,
        stockfish_executable=namespace.stockfish_executable,
        match_nodes=namespace.match_nodes,
        opening_pairs=namespace.opening_pairs,
        device_id=namespace.device_id,
        output=namespace.output,
    )
    required_paths = (
        arguments.experiment,
        arguments.run_directory,
        arguments.opening_manifest,
        arguments.stockfish_executable,
    )
    if not all(path.exists() for path in required_paths):
        raise ValueError('Experiment, run directory, openings, and Stockfish executable must exist.')
    if arguments.output.exists():
        raise ValueError(f'Gauntlet output already exists: {arguments.output}')
    if arguments.output.parent and not arguments.output.parent.is_dir():
        raise ValueError(f'Gauntlet output directory does not exist: {arguments.output.parent}')
    positive_values = (
        arguments.match_nodes,
        arguments.opening_pairs,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError('Match nodes and opening pairs must be positive.')
    if arguments.checkpoint_generation < 0 or arguments.device_id < 0:
        raise ValueError('Checkpoint generation and device ID must be nonnegative.')
    return arguments


def main() -> None:
    print(run_gauntlet(parse_arguments()).model_dump_json(indent=2))


if __name__ == '__main__':
    main()
