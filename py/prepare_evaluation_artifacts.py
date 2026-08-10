from __future__ import annotations

import argparse
import os
from pathlib import Path
import site
import subprocess

from src.evaluation.preparation import prepare_evaluation_artifacts
from src.experiment.configuration import load_experiment_configuration
from src.games.composition import create_game_implementation


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare immutable evaluation inputs without starting training.')
    parser.add_argument('--run-config', required=True, type=Path)
    return parser.parse_args()


def repository_revision(repository_root: Path) -> str:
    result = subprocess.run(
        ('git', 'rev-parse', 'HEAD'),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def configure_python_nvidia_libraries() -> None:
    library_directories = tuple(
        path
        for site_package_directory in site.getsitepackages()
        for path in sorted((Path(site_package_directory) / 'nvidia').glob('*/lib'))
        if path.is_dir()
    )
    if not library_directories:
        return
    existing = os.environ.get('LD_LIBRARY_PATH')
    configured = os.pathsep.join(str(path) for path in library_directories)
    os.environ['LD_LIBRARY_PATH'] = configured if existing is None else f'{configured}{os.pathsep}{existing}'


def main() -> None:
    arguments = parse_arguments()
    repository_root = Path(__file__).resolve().parent.parent
    configure_python_nvidia_libraries()
    experiment = load_experiment_configuration(arguments.run_config.resolve())
    artifacts = prepare_evaluation_artifacts(
        experiment,
        create_game_implementation(experiment),
        repository_root,
        repository_revision(repository_root),
    )
    print(f'Dataset: {artifacts.dataset_path}')
    print(f'Dataset manifest: {artifacts.dataset_manifest_path}')
    print(f'Openings: {artifacts.opening_manifest_path}')


if __name__ == '__main__':
    main()
