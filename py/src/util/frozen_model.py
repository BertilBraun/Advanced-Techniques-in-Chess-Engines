from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

JsonValue: TypeAlias = str | int | float | bool | None | list['JsonValue'] | dict[str, 'JsonValue']


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    def validated_copy(self, *, update: Mapping[str, JsonValue]) -> Self:
        payload = self.model_dump(mode='json')
        payload.update(update)
        return self.__class__.model_validate(payload)
