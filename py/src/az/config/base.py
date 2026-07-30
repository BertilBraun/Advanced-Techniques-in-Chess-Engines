from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, allow_inf_nan=False)


class DeterminismMode(str, Enum):
    STRICT_SINGLE_THREAD = 'strict_single_thread'
    SEEDED_CONCURRENT = 'seeded_concurrent'


Sha256 = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
GitRevision = Annotated[str, Field(pattern=r'^[0-9a-f]{40}$')]
