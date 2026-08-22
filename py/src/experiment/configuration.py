"""Load the discriminated union of concrete game experiment configurations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, TypeAlias

import yaml
from pydantic import Field, JsonValue, TypeAdapter
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.go.configuration import GoExperimentConfiguration
from src.util.atomic_file import write_text_atomically

ExperimentConfiguration: TypeAlias = Annotated[
    ChessExperimentConfiguration | GoExperimentConfiguration,
    Field(discriminator='game'),
]
EXPERIMENT_CONFIGURATION_ADAPTER = TypeAdapter(ExperimentConfiguration)

JSON_MAPPING_ADAPTER = TypeAdapter(dict[str, JsonValue])


def load_experiment_configuration(path: Path) -> ExperimentConfiguration:
    resolved, _ = _load_experiment_mapping(path.resolve(), ())
    return EXPERIMENT_CONFIGURATION_ADAPTER.validate_python(resolved)


def load_experiment_configuration_json(payload: str) -> ExperimentConfiguration:
    return EXPERIMENT_CONFIGURATION_ADAPTER.validate_json(payload)


def experiment_configuration_source_paths(path: Path) -> tuple[Path, ...]:
    _, source_paths = _load_experiment_mapping(path.resolve(), ())
    return source_paths


def experiment_configuration_sha256(configuration: ExperimentConfiguration) -> str:
    return hashlib.sha256(configuration.model_dump_json().encode('utf-8')).hexdigest()


def _load_experiment_mapping(
    path: Path,
    inheritance_chain: tuple[Path, ...],
) -> tuple[dict[str, JsonValue], tuple[Path, ...]]:
    if path in inheritance_chain:
        cycle = ' -> '.join(str(item) for item in (*inheritance_chain, path))
        raise ValueError(f'Experiment configuration inheritance contains a cycle: {cycle}')
    payload = path.read_text(encoding='utf-8')
    parsed = yaml.safe_load(payload) if path.suffix.casefold() in {'.yaml', '.yml'} else json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError(f'Experiment file must contain a mapping: {path}')
    mapping = JSON_MAPPING_ADAPTER.validate_python(parsed)
    extends = mapping.pop('extends', None)
    if extends is None:
        return mapping, (path,)
    if not isinstance(extends, str) or not extends.strip():
        raise ValueError(f'Experiment extends must be a nonempty path string: {path}')
    base_path = Path(extends)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base, source_paths = _load_experiment_mapping(base_path.resolve(), (*inheritance_chain, path))
    return _merge_experiment_mappings(base, mapping), (*source_paths, path)


def _merge_experiment_mappings(
    base: dict[str, JsonValue],
    override: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    merged = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        discriminator_changed = (
            isinstance(base_value, dict)
            and isinstance(override_value, dict)
            and 'kind' in override_value
            and base_value.get('kind') != override_value['kind']
        )
        if isinstance(base_value, dict) and isinstance(override_value, dict) and not discriminator_changed:
            merged[key] = _merge_experiment_mappings(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


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
