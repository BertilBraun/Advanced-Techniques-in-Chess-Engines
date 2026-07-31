from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel

from src.az.config.canonical import canonical_json
from src.az.config.models import ResolvedRunConfiguration
from src.az.config.resolution import (
    AuthoringRunConfiguration,
    resolve_configuration,
    validate_authoring_configuration_json,
)
from src.az.config.root import validate_resolved_configuration_json


def model_sha256(model: BaseModel) -> str:
    return hashlib.sha256(canonical_json(model).encode("utf-8")).hexdigest()


def load_authoring_configuration(path: Path) -> AuthoringRunConfiguration:
    return validate_authoring_configuration_json(path.read_text(encoding="utf-8"))


def load_resolved_configuration(path: Path) -> ResolvedRunConfiguration:
    return validate_resolved_configuration_json(path.read_text(encoding="utf-8"))


def write_resolved_configuration(
    path: Path, configuration: ResolvedRunConfiguration
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(configuration.model_dump_json(indent=2) + "\n", encoding="utf-8")


def resolve_file(path: Path) -> ResolvedRunConfiguration:
    return resolve_configuration(load_authoring_configuration(path))
