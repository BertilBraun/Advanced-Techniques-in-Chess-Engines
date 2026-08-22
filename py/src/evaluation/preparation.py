from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.evaluation.contracts import AnyEvaluationDatasetManifest, AnyOpeningSuiteManifest
from src.evaluation.dataset import (
    build_evaluation_dataset,
    build_katago_book_evaluation_dataset,
    dataset_manifest_path,
)
from src.evaluation.openings import (
    build_chess_book_opening_suite,
    build_katago_book_opening_suite,
    build_opening_suite,
)
from src.experiment.configuration import ExperimentConfiguration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.stockfish import StockfishClient
from src.games.chess.training import ChessImplementation
from src.games.composition import ConfiguredGame
from src.games.go.configuration import GoExperimentConfiguration
from src.games.go.katago import GoEvaluationArtifactMetadata, KataGoClient
from src.games.go.training import GoImplementation


@dataclass(frozen=True)
class PreparedEvaluationArtifacts:
    dataset_path: Path
    dataset_manifest_path: Path
    dataset_manifest: AnyEvaluationDatasetManifest
    opening_manifest_path: Path
    opening_manifest: AnyOpeningSuiteManifest


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
    match experiment, game, evaluation.openings.source, evaluation.dataset.source:
        case GoExperimentConfiguration(), GoImplementation(), opening_source, dataset_source if (
            opening_source.kind == 'katago_book' and dataset_source.kind == 'katago_book'
        ):
            metadata = GoEvaluationArtifactMetadata(game.state)
            opening_manifest = build_katago_book_opening_suite(
                opening_path,
                _resolve_source_path(source_root, opening_source.selection.export_path),
                evaluation.openings,
                game.state,
                metadata,
                source_revision,
            )
            dataset_manifest = build_katago_book_evaluation_dataset(
                dataset_path,
                _resolve_source_path(source_root, dataset_source.selection.export_path),
                evaluation.dataset,
                game.state,
                metadata,
                source_revision,
            )
            return PreparedEvaluationArtifacts(
                dataset_path=dataset_path,
                dataset_manifest_path=dataset_manifest_path(dataset_path),
                dataset_manifest=dataset_manifest,
                opening_manifest_path=opening_path,
                opening_manifest=opening_manifest,
            )
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
        match evaluation.openings.source:
            case source if source.kind == 'engine_beam':
                opening_manifest = build_opening_suite(
                    opening_path,
                    evaluation.openings,
                    game.state,
                    engine,
                    source_revision,
                )
            case source if source.kind == 'chess_book':
                if not isinstance(game, ChessImplementation):
                    raise ValueError('Chess book openings require the chess game implementation.')
                opening_manifest = build_chess_book_opening_suite(
                    opening_path,
                    _resolve_source_path(source_root, source.selection_path),
                    evaluation.openings,
                    game.state,
                    engine,
                    source_revision,
                )
            case source:
                opening_manifest = build_katago_book_opening_suite(
                    opening_path,
                    _resolve_source_path(source_root, source.selection.export_path),
                    evaluation.openings,
                    game.state,
                    engine,
                    source_revision,
                )
        match evaluation.dataset.source:
            case source if source.kind == 'engine_self_play':
                dataset_manifest = build_evaluation_dataset(
                    dataset_path,
                    evaluation.dataset,
                    game.state,
                    engine,
                    source_revision,
                )
            case source:
                dataset_manifest = build_katago_book_evaluation_dataset(
                    dataset_path,
                    _resolve_source_path(source_root, source.selection.export_path),
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
