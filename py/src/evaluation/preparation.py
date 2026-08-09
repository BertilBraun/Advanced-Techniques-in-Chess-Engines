from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.evaluation.dataset import build_evaluation_dataset, dataset_manifest_path
from src.evaluation.openings import build_opening_suite
from src.evaluation.contracts import EvaluationDatasetManifest, OpeningSuiteManifest
from src.experiment.configuration import ExperimentConfiguration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.stockfish import StockfishClient
from src.games.chess.training import ChessImplementation
from src.games.composition import ConfiguredGame
from src.games.go.configuration import GoExperimentConfiguration
from src.games.go.katago import KataGoClient
from src.games.go.training import GoImplementation


@dataclass(frozen=True)
class PreparedEvaluationArtifacts:
    dataset_path: Path
    dataset_manifest_path: Path
    dataset_manifest: EvaluationDatasetManifest
    opening_manifest_path: Path
    opening_manifest: OpeningSuiteManifest


def _resolve_source_path(source_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else source_root / path


def prepare_evaluation_artifacts(
    experiment: ExperimentConfiguration,
    game: ConfiguredGame,
    source_root: Path,
    source_revision: str,
) -> PreparedEvaluationArtifacts:
    evaluation = experiment.evaluation
    dataset_path = _resolve_source_path(source_root, evaluation.dataset.path)
    opening_path = _resolve_source_path(source_root, evaluation.openings.path)
    match experiment, game, evaluation.engine:
        case ChessExperimentConfiguration(), ChessImplementation(), engine_configuration:
            if engine_configuration.kind != 'stockfish':
                raise ValueError('Chess evaluation requires Stockfish engine configuration.')
            engine = StockfishClient(
                engine_configuration,
                game.state,
                _resolve_source_path(source_root, engine_configuration.executable_path),
            )
        case GoExperimentConfiguration(), GoImplementation(), engine_configuration:
            if engine_configuration.kind != 'katago':
                raise ValueError('Go evaluation requires KataGo engine configuration.')
            engine = KataGoClient(
                engine_configuration,
                game.state,
                _resolve_source_path(source_root, engine_configuration.executable_path),
                _resolve_source_path(source_root, engine_configuration.model_path),
                _resolve_source_path(source_root, engine_configuration.analysis_configuration_path),
            )
        case _:
            raise ValueError('Experiment and game implementations do not match.')
    try:
        opening_manifest = build_opening_suite(
            opening_path,
            evaluation.openings,
            game.state,
            engine,
            source_revision,
        )
        dataset_manifest = build_evaluation_dataset(
            dataset_path,
            evaluation.dataset,
            game.state,
            engine,
            source_revision,
        )
    finally:
        engine.close()
    return PreparedEvaluationArtifacts(
        dataset_path=dataset_path,
        dataset_manifest_path=dataset_manifest_path(dataset_path),
        dataset_manifest=dataset_manifest,
        opening_manifest_path=opening_path,
        opening_manifest=opening_manifest,
    )
