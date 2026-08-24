from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, PlainSerializer
from typing_extensions import Self

JsonValue: TypeAlias = str | int | float | bool | None | list['JsonValue'] | dict[str, 'JsonValue']

# Configuration hashes must be stable across host operating systems, so Path fields in
# configuration models serialise with forward slashes regardless of the host separator.
ConfigurationPath: TypeAlias = Annotated[Path, PlainSerializer(lambda path: path.as_posix(), return_type=str)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    def validated_copy(self, *, update: Mapping[str, JsonValue]) -> Self:
        payload = self.model_dump(mode='json')
        payload.update(update)
        return self.__class__.model_validate(payload)
