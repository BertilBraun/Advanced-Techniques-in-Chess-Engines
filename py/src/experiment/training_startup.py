from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from src.experiment.configuration import (
    ExperimentConfiguration,
    load_experiment_configuration,
    write_resolved_experiment,
)
from src.experiment.run import ExperimentRunManifest, prepare_experiment_training_run
from src.util.tensorboard import configure_tensorboard_run_directory, get_run_id


@dataclass(frozen=True)
class PreparedTrainingStartup:
    experiment: ExperimentConfiguration
    manifest: ExperimentRunManifest
    run_id: int
    started_at: float


def prepare_training_startup(
    run_configuration_path: Path,
    expected_source_revision: str,
    approval_path: Path,
) -> PreparedTrainingStartup:
    experiment = load_experiment_configuration(run_configuration_path.resolve())
    started_at = monotonic()
    manifest = prepare_experiment_training_run(experiment, expected_source_revision, approval_path)
    write_resolved_experiment(Path(experiment.training.save_path) / 'resolved-experiment.json', experiment)
    configure_tensorboard_run_directory(experiment.run.tensorboard_run_directory)
    return PreparedTrainingStartup(
        experiment=experiment,
        manifest=manifest,
        run_id=get_run_id(),
        started_at=started_at,
    )
