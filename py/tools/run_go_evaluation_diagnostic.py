from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.configuration import (
    EvaluationSearchConfiguration,
    KataGoEvaluationDefinition,
)
from src.evaluation.contracts import OpeningSuiteManifest
from src.evaluation.go_diagnostic import (
    GoEvaluationDiagnosticResult,
    KataGoDiagnosticEngine,
    KataGoDiagnosticSelector,
    ProjectModelDiagnosticEngine,
    ProjectModelDiagnosticSelector,
    render_go_diagnostic_analysis,
    run_go_diagnostic_match,
    validate_deterministic_identity,
    write_go_diagnostic_match,
    write_go_diagnostic_result,
)
from src.experiment.configuration import load_experiment_configuration
from src.games.go.configuration import GoExperimentConfiguration
from src.games.go.katago import KataGoClient
from src.games.go.training import GoImplementation
from src.training.checkpoint import CheckpointReference
from src.util.atomic_file import write_text_atomically
from src.util.hashing import file_sha256
from src.util.provenance import read_source_revision

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Arguments:
    experiment: Path
    baseline_run: Path
    checkpoint_generation: int
    opening_manifest: Path
    output_directory: Path
    identity_openings: int
    full_openings: int
    project_first_device: int
    project_second_device: int
    katago_equal_device: int
    katago_16_device: int
    katago_128_device: int


def _project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _model_search(configuration: GoExperimentConfiguration) -> EvaluationSearchConfiguration:
    for definition in configuration.evaluation.definitions:
        if isinstance(definition, KataGoEvaluationDefinition):
            return definition.search
    raise ValueError('Go diagnostic requires a configured KataGo evaluation search.')


def _katago_selector(
    configuration: GoExperimentConfiguration,
    game: GoImplementation,
    engine: KataGoDiagnosticEngine,
    visible_device: int,
) -> KataGoDiagnosticSelector:
    external = configuration.evaluation.engine
    if external.kind != 'katago':
        raise ValueError('Go diagnostic requires KataGo engine configuration.')
    client = KataGoClient(
        external,
        game.state,
        _project_path(external.executable_path),
        _project_path(external.model_path),
        _project_path(external.analysis_configuration_path),
        visible_cuda_device=visible_device,
    )
    return KataGoDiagnosticSelector(engine, client)


def run_diagnostic(arguments: Arguments) -> GoEvaluationDiagnosticResult:
    started_at = time.monotonic()
    loaded = load_experiment_configuration(arguments.experiment)
    if not isinstance(loaded, GoExperimentConfiguration):
        raise ValueError('Go evaluation diagnostic requires a Go experiment.')
    game = GoImplementation(loaded)
    checkpoint = CheckpointReference.load_for_inference(
        arguments.baseline_run,
        arguments.checkpoint_generation,
    )
    openings = OpeningSuiteManifest.model_validate_json(arguments.opening_manifest.read_text(encoding='utf-8'))
    if openings.game != 'go':
        raise ValueError('Go evaluation diagnostic requires a Go opening manifest.')
    arguments.output_directory.mkdir(parents=True, exist_ok=False)
    model_engine = ProjectModelDiagnosticEngine(
        kind='project_model',
        engine_id=f'baseline-generation-{checkpoint.generation}',
        checkpoint=checkpoint,
        search=_model_search(loaded),
    )
    first_model = ProjectModelDiagnosticSelector(model_engine, game, arguments.project_first_device)
    second_model = ProjectModelDiagnosticSelector(model_engine, game, arguments.project_second_device)
    identity = run_go_diagnostic_match(
        'baseline-deterministic-identity',
        game.state,
        openings,
        arguments.identity_openings,
        first_model,
        second_model,
        loaded.go.rules.maximum_moves,
        loaded.training.random_seed,
        loaded.evaluation.bootstrap_samples,
    )
    validate_deterministic_identity(identity)
    write_go_diagnostic_match(identity, arguments.output_directory / '01-model-identity.json')

    katago_16 = KataGoDiagnosticEngine(kind='katago', engine_id='katago-16', maximum_visits=16)
    equal_selector = _katago_selector(
        loaded,
        game,
        katago_16,
        arguments.katago_equal_device,
    )
    try:
        equal = run_go_diagnostic_match(
            'katago-16-identity',
            game.state,
            openings,
            arguments.full_openings,
            equal_selector,
            equal_selector,
            loaded.go.rules.maximum_moves,
            loaded.training.random_seed + 100_000,
            loaded.evaluation.bootstrap_samples,
        )
    finally:
        equal_selector.close()
    write_go_diagnostic_match(equal, arguments.output_directory / '02-katago-16-identity.json')

    katago_128 = KataGoDiagnosticEngine(kind='katago', engine_id='katago-128', maximum_visits=128)
    selector_16 = _katago_selector(loaded, game, katago_16, arguments.katago_16_device)
    selector_128 = _katago_selector(loaded, game, katago_128, arguments.katago_128_device)
    try:
        visits = run_go_diagnostic_match(
            'katago-16-versus-128',
            game.state,
            openings,
            arguments.full_openings,
            selector_16,
            selector_128,
            loaded.go.rules.maximum_moves,
            loaded.training.random_seed + 200_000,
            loaded.evaluation.bootstrap_samples,
        )
    finally:
        selector_16.close()
        selector_128.close()
    write_go_diagnostic_match(visits, arguments.output_directory / '03-katago-16-versus-128.json')

    result = GoEvaluationDiagnosticResult(
        source_revision=read_source_revision().commit,
        experiment_path=arguments.experiment.resolve(),
        opening_manifest_path=arguments.opening_manifest.resolve(),
        opening_manifest_sha256=file_sha256(arguments.opening_manifest),
        baseline_run_path=arguments.baseline_run.resolve(),
        evaluated_checkpoint=checkpoint,
        matches=(identity, equal, visits),
        duration_seconds=time.monotonic() - started_at,
    )
    write_go_diagnostic_result(result, arguments.output_directory / 'result.json')
    write_text_atomically(
        arguments.output_directory / 'analysis.md',
        render_go_diagnostic_analysis(result) + '\n',
    )
    return result


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Run the focused paired Go evaluation diagnostic.')
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--baseline-run', required=True, type=Path)
    parser.add_argument('--checkpoint-generation', required=True, type=int)
    parser.add_argument('--opening-manifest', required=True, type=Path)
    parser.add_argument('--output-directory', required=True, type=Path)
    parser.add_argument('--identity-openings', default=5, type=int)
    parser.add_argument('--full-openings', default=50, type=int)
    parser.add_argument('--project-first-device', default=0, type=int)
    parser.add_argument('--project-second-device', default=1, type=int)
    parser.add_argument('--katago-equal-device', default=2, type=int)
    parser.add_argument('--katago-16-device', default=2, type=int)
    parser.add_argument('--katago-128-device', default=3, type=int)
    namespace = parser.parse_args()
    arguments = Arguments(
        experiment=namespace.experiment,
        baseline_run=namespace.baseline_run,
        checkpoint_generation=namespace.checkpoint_generation,
        opening_manifest=namespace.opening_manifest,
        output_directory=namespace.output_directory,
        identity_openings=namespace.identity_openings,
        full_openings=namespace.full_openings,
        project_first_device=namespace.project_first_device,
        project_second_device=namespace.project_second_device,
        katago_equal_device=namespace.katago_equal_device,
        katago_16_device=namespace.katago_16_device,
        katago_128_device=namespace.katago_128_device,
    )
    paths = (arguments.experiment, arguments.baseline_run, arguments.opening_manifest)
    if not all(path.exists() for path in paths):
        raise ValueError('Experiment, baseline run, and opening manifest must exist.')
    counts = (arguments.identity_openings, arguments.full_openings)
    devices = (
        arguments.project_first_device,
        arguments.project_second_device,
        arguments.katago_equal_device,
        arguments.katago_16_device,
        arguments.katago_128_device,
    )
    if any(count <= 0 for count in counts):
        raise ValueError('Diagnostic opening counts must be positive.')
    if any(device < 0 for device in devices) or arguments.checkpoint_generation < 0:
        raise ValueError('Diagnostic devices and checkpoint generation must be nonnegative.')
    return arguments


def main() -> None:
    result = run_diagnostic(parse_arguments())
    print(result.model_dump_json(indent=2))


if __name__ == '__main__':
    main()
