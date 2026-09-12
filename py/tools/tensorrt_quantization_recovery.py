from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field
from src.util.frozen_model import FrozenModel


class DirectQuantizationOutput(FrozenModel):
    kind: Literal['direct'] = 'direct'


class RecoveredModelOptAutotuneOutput(FrozenModel):
    kind: Literal['recovered_modelopt_autotune'] = 'recovered_modelopt_autotune'
    condition: Literal['remove_partial_input_qdq_index_error'] = 'remove_partial_input_qdq_index_error'
    exception_type: Literal['IndexError'] = 'IndexError'
    message: str = Field(min_length=1)


QuantizationOutputIdentity = Annotated[
    DirectQuantizationOutput | RecoveredModelOptAutotuneOutput,
    Field(discriminator='kind'),
]


def identify_modelopt_autotune_recovery(error: Exception) -> RecoveredModelOptAutotuneOutput | None:
    if not isinstance(error, IndexError) or str(error) != 'list index out of range':
        return None
    traceback = error.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_name == 'remove_partial_input_qdq':
            return RecoveredModelOptAutotuneOutput(message=str(error))
        traceback = traceback.tb_next
    return None
