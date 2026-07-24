from __future__ import annotations

import argparse
import copy
from pathlib import Path
import subprocess
import time

from pydantic import BaseModel, ConfigDict, Field
import torch.multiprocessing as multiprocessing

from src.cluster.EvaluationProcess import EvaluationProcess
from src.cluster.EvaluationProcess import EvaluationRunProvenance
from src.experiment.run_configuration import (
    apply_run_configuration,
    load_run_configuration,
)
from src.settings import TRAINING_ARGS


class EvaluationSuiteBenchmarkResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    source_revision: str = Field(pattern=r'^[0-9a-f]{40}$')
    evaluated_model_source_revision: str = Field(pattern=r'^[0-9a-f]{40}$')
    stockfish_binary_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    run_configuration_path: str
    model_version: int = Field(ge=0)
    tensorboard_run_id: int = Field(ge=0)
    games_per_match: int = Field(gt=0)
    searches_per_turn: int = Field(gt=0)
    maximum_concurrent_tasks: int = Field(gt=0)
    elapsed_seconds: float = Field(gt=0)
    raw_results_directory: str
    model_directory: str


def _source_revision(source_root: Path) -> str:
    completed = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description='Benchmark one complete configured evaluation suite.')
    parser.add_argument('--run-config', required=True, type=Path)
    parser.add_argument('--model-version', required=True, type=int)
    parser.add_argument('--model-directory', required=True, type=Path)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--stockfish-binary-sha256', required=True)
    parser.add_argument('--tensorboard-run-id', required=True, type=int)
    parser.add_argument('--raw-results-directory', required=True, type=Path)
    parser.add_argument('--output-path', required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.model_version < 0 or arguments.tensorboard_run_id < 0:
        raise ValueError('Model version and TensorBoard run ID must be nonnegative.')
    if len(arguments.source_revision) != 40 or any(
        character not in '0123456789abcdef' for character in arguments.source_revision
    ):
        raise ValueError('Source revision must be a lowercase 40-character Git hash.')
    if len(arguments.stockfish_binary_sha256) != 64 or any(
        character not in '0123456789abcdef' for character in arguments.stockfish_binary_sha256
    ):
        raise ValueError('Stockfish SHA-256 must be a lowercase 64-character hash.')
    if arguments.raw_results_directory.exists():
        raise ValueError(f'Raw evaluation directory already exists: {arguments.raw_results_directory}')
    if arguments.output_path.exists():
        raise ValueError(f'Evaluation benchmark output already exists: {arguments.output_path}')

    configuration = load_run_configuration(arguments.run_config)
    evaluation_protocol = configuration.evaluation_protocol.model_copy(
        update={'raw_results_subdirectory': str(arguments.raw_results_directory.resolve())}
    )
    isolated_configuration = configuration.model_copy(update={'evaluation_protocol': evaluation_protocol})
    training_arguments = copy.deepcopy(TRAINING_ARGS)
    apply_run_configuration(training_arguments, isolated_configuration)
    training_arguments.save_path = str(arguments.model_directory.resolve())
    evaluation_arguments = training_arguments.evaluation
    if evaluation_arguments is None:
        raise ValueError('Evaluation benchmark requires an evaluation configuration.')

    started_at = time.perf_counter()
    EvaluationProcess(training_arguments).run(
        arguments.tensorboard_run_id,
        arguments.model_version,
        metrics_step=arguments.model_version,
        provenance=EvaluationRunProvenance(
            source_revision=arguments.source_revision,
            stockfish_binary_sha256=arguments.stockfish_binary_sha256,
        ),
    )
    elapsed_seconds = time.perf_counter() - started_at

    result = EvaluationSuiteBenchmarkResult(
        source_revision=_source_revision(Path(__file__).resolve().parents[2]),
        evaluated_model_source_revision=arguments.source_revision,
        stockfish_binary_sha256=arguments.stockfish_binary_sha256,
        run_configuration_path=str(arguments.run_config.resolve()),
        model_version=arguments.model_version,
        tensorboard_run_id=arguments.tensorboard_run_id,
        games_per_match=evaluation_arguments.num_games,
        searches_per_turn=evaluation_arguments.num_searches_per_turn,
        maximum_concurrent_tasks=evaluation_arguments.max_concurrent_tasks,
        elapsed_seconds=elapsed_seconds,
        raw_results_directory=str(arguments.raw_results_directory.resolve()),
        model_directory=str(arguments.model_directory.resolve()),
    )
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_path.write_text(result.model_dump_json(indent=2) + '\n', encoding='utf-8')
    print(result.model_dump_json(indent=2))


if __name__ == '__main__':
    main()
