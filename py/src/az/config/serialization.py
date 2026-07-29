from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

from src.az.config.models import ResolvedRunConfiguration
from src.az.config.resolution import AuthoringRunConfiguration, resolve_configuration


def canonical_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode='json'),
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True,
    )


def model_sha256(model: BaseModel) -> str:
    return hashlib.sha256(canonical_json(model).encode('utf-8')).hexdigest()


def load_authoring_configuration(path: Path) -> AuthoringRunConfiguration:
    return AuthoringRunConfiguration.model_validate_json(path.read_text(encoding='utf-8'))


def load_resolved_configuration(path: Path) -> ResolvedRunConfiguration:
    return ResolvedRunConfiguration.model_validate_json(path.read_text(encoding='utf-8'))


def write_resolved_configuration(path: Path, configuration: ResolvedRunConfiguration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(configuration.model_dump_json(indent=2) + '\n', encoding='utf-8')


def resolve_file(path: Path) -> ResolvedRunConfiguration:
    return resolve_configuration(load_authoring_configuration(path))
