from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer
from typing_extensions import Self

JsonValue: TypeAlias = str | int | float | bool | None | list['JsonValue'] | dict[str, 'JsonValue']


def _normalize_configuration_path(path: str | Path) -> str:
    return str(path).replace('\\', '/')


# Configuration hashes must be stable across host operating systems, so paths accept either separator and serialize
# with forward slashes.
ConfigurationPath: TypeAlias = Annotated[
    Path,
    BeforeValidator(_normalize_configuration_path),
    PlainSerializer(lambda path: path.as_posix(), return_type=str),
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    def validated_copy(self, *, update: Mapping[str, JsonValue]) -> Self:
        payload = self.model_dump(mode='json')
        payload.update(update)
        return self.__class__.model_validate(payload)
