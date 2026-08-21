from __future__ import annotations

import os
from pathlib import Path
import signal
import time
import traceback
from types import FrameType
from typing import TYPE_CHECKING

from src.evaluation.configuration import KataGoEngineConfiguration, StockfishEngineConfiguration
from src.evaluation.contracts import (
    EvaluationFailurePhase,
    EVALUATION_JOB_ADAPTER,
    EvaluationResult,
    FailedEvaluationResult,
    FixedDatasetEvaluationJob,
    MatchEvaluationJob,
    OPENING_SUITE_MANIFEST_ADAPTER,
)
from src.experiment.configuration import ExperimentConfiguration, load_experiment_configuration_json
from src.util.atomic_file import write_text_atomically

if TYPE_CHECKING:
    from src.games.chess.stockfish import StockfishFixedNodesMatchEngine, StockfishMatchEngine
    from src.games.composition import ConfiguredGame
    from src.games.go.katago import KataGoMatchEngine


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class EvaluationTermination(Exception):
    pass


def _terminate_evaluation(signum: int, frame: FrameType | None) -> None:
    raise EvaluationTermination(f'Evaluation process received signal {signum}.')


def resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def write_evaluation_result(result: EvaluationResult, path: Path) -> None:
    write_text_atomically(path, result.model_dump_json(indent=2) + '\n')


def run_evaluation_job(experiment_json: str, job_json: str) -> None:
    experiment = load_experiment_configuration_json(experiment_json)
    job = EVALUATION_JOB_ADAPTER.validate_json(job_json)
    if os.name == 'posix':
        os.setpgrp()
    started_at = time.monotonic()
    signal.signal(signal.SIGTERM, _terminate_evaluation)
    try:
        import torch

        if experiment.training.topology.trainer.device_type == 'cuda':
            torch.cuda.set_device(job.device_id)
        from src.games.composition import create_game_implementation

        game = create_game_implementation(experiment)
        match job:
            case FixedDatasetEvaluationJob():
                from src.evaluation.dataset import evaluate_fixed_dataset

                result = evaluate_fixed_dataset(
                    job,
                    game.state,
                    resolve_project_path(experiment.evaluation.dataset.path),
                    experiment.training.topology.trainer.device_type,
                )
            case MatchEvaluationJob():
                from src.evaluation.match import run_match

                openings = OPENING_SUITE_MANIFEST_ADAPTER.validate_json(
                    resolve_project_path(experiment.evaluation.openings.path).read_text(encoding='utf-8')
                )
                external_engine = _create_external_match_engine(experiment, job, game)
                try:
                    result = run_match(
                        job,
                        game,
                        openings,
                        experiment.evaluation.bootstrap_samples,
                        external_engine,
                        experiment.training.topology.trainer.device_type,
                    )
                finally:
                    if external_engine is not None:
                        external_engine.close()
        write_evaluation_result(result, job.result_path)
    except Exception as error:
        traceback_path = job.result_path.with_suffix('.traceback.txt')
        write_text_atomically(traceback_path, traceback.format_exc())
        failure = FailedEvaluationResult(
            kind='failed',
            job=job,
            phase=EvaluationFailurePhase.EXECUTION,
            message=str(error) or error.__class__.__name__,
            exit_code=None,
            traceback_path=traceback_path,
            duration_seconds=time.monotonic() - started_at,
        )
        write_evaluation_result(failure, job.result_path)


def _create_external_match_engine(
    experiment: ExperimentConfiguration,
    job: MatchEvaluationJob,
    game: ConfiguredGame,
) -> StockfishFixedNodesMatchEngine | StockfishMatchEngine | KataGoMatchEngine | None:
    from src.games.chess.stockfish import StockfishFixedNodesMatchEngine, StockfishMatchEngine
    from src.games.go.katago import KataGoMatchEngine

    match job.opponent.kind:
        case 'random' | 'checkpoint':
            return None
        case 'stockfish':
            from src.games.chess.stockfish import StockfishClient

            engine_configuration = experiment.evaluation.engine
            if not isinstance(engine_configuration, StockfishEngineConfiguration):
                raise ValueError('Stockfish job requires Stockfish engine configuration.')
            client = StockfishClient(
                engine_configuration,
                game.state,
                resolve_project_path(engine_configuration.executable_path),
            )
            return StockfishMatchEngine(client, job.opponent.skill_level)
        case 'stockfish_fixed_nodes':
            from src.evaluation.contracts import StockfishFixedNodesOpponent
            from src.games.chess.stockfish import StockfishClient

            engine_configuration = experiment.evaluation.engine
            if not isinstance(engine_configuration, StockfishEngineConfiguration):
                raise ValueError('Stockfish fixed-node job requires Stockfish engine configuration.')
            if not isinstance(job.opponent, StockfishFixedNodesOpponent):
                raise ValueError('Stockfish fixed-node job requires a fixed-node opponent.')
            client = StockfishClient(
                engine_configuration,
                game.state,
                resolve_project_path(engine_configuration.executable_path),
            )
            return StockfishFixedNodesMatchEngine(client, job.opponent.nodes)
        case 'katago':
            from src.games.go.katago import KataGoClient

            engine_configuration = experiment.evaluation.engine
            if not isinstance(engine_configuration, KataGoEngineConfiguration):
                raise ValueError('KataGo job requires KataGo engine configuration.')
            client = KataGoClient(
                engine_configuration,
                game.state,
                resolve_project_path(engine_configuration.executable_path),
                resolve_project_path(engine_configuration.model_path),
                resolve_project_path(engine_configuration.analysis_configuration_path),
                visible_cuda_device=job.device_id,
            )
            return KataGoMatchEngine(client, job.opponent.maximum_visits)
