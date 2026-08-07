"""Load the discriminated union of concrete game experiment configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, TypeAlias

import yaml
from pydantic import Field, TypeAdapter

from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.go.configuration import GoExperimentConfiguration
from src.util.atomic_file import write_text_atomically


ExperimentConfiguration: TypeAlias = Annotated[
    ChessExperimentConfiguration | GoExperimentConfiguration,
    Field(discriminator='game'),
]


def load_experiment_configuration(path: Path) -> ExperimentConfiguration:
    payload = path.read_text(encoding='utf-8')
    parsed = yaml.safe_load(payload) if path.suffix.casefold() in {'.yaml', '.yml'} else json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError(f'Experiment file must contain a mapping: {path}')
    return TypeAdapter(ExperimentConfiguration).validate_python(parsed)


def load_chess_experiment_configuration(path: Path) -> ChessExperimentConfiguration:
    configuration = load_experiment_configuration(path)
    if not isinstance(configuration, ChessExperimentConfiguration):
        raise ValueError(f'Expected a chess experiment configuration: {path}')
    return configuration


def validate_experiment_queue(paths: tuple[Path, ...]) -> tuple[ExperimentConfiguration, ...]:
    if not paths:
        raise ValueError('Experiment queue validation requires at least one configuration path.')
    return tuple(load_experiment_configuration(path) for path in paths)


def write_resolved_experiment(path: Path, configuration: ExperimentConfiguration) -> None:
    write_text_atomically(path, configuration.model_dump_json(indent=2) + '\n')


def write_resolved_chess_experiment(path: Path, configuration: ChessExperimentConfiguration) -> None:
    write_resolved_experiment(path, configuration)
