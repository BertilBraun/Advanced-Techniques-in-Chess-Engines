from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from src.training.network import NetworkDefinition
from src.util.frozen_model import FrozenModel


class ParameterBand(str, Enum):
    ONE_MILLION = '1m'
    FOUR_MILLION = '4m'
    NINE_MILLION = '9m'


class ArchitectureCatalogEntry(FrozenModel):
    model_id: str = Field(min_length=1)
    parameter_band: ParameterBand
    expected_training_parameters: int = Field(gt=0)
    definition: NetworkDefinition


class ArchitectureCatalog(FrozenModel):
    schema_version: Literal[1]
    models: tuple[ArchitectureCatalogEntry, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_unique_model_ids(self) -> ArchitectureCatalog:
        model_ids = tuple(model.model_id for model in self.models)
        if len(set(model_ids)) != len(model_ids):
            raise ValueError('Architecture catalog model IDs must be unique.')
        return self


def load_architecture_catalog(path: Path) -> ArchitectureCatalog:
    return ArchitectureCatalog.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')))
