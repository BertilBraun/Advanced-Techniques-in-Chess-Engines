from __future__ import annotations

from typing import TYPE_CHECKING

from src.self_play.configuration import SdpaBackend

if TYPE_CHECKING:
    from AlphaZeroCpp import SdpaBackend as NativeSdpaBackend


def native_sdpa_backend(backend: SdpaBackend) -> NativeSdpaBackend:
    from AlphaZeroCpp import SdpaBackend as NativeSdpaBackend

    match backend:
        case SdpaBackend.AUTOMATIC:
            return NativeSdpaBackend.AUTOMATIC
        case SdpaBackend.FLASH:
            return NativeSdpaBackend.FLASH
        case SdpaBackend.MEMORY_EFFICIENT:
            return NativeSdpaBackend.MEMORY_EFFICIENT
        case SdpaBackend.MATH:
            return NativeSdpaBackend.MATH
        case SdpaBackend.CUDNN:
            return NativeSdpaBackend.CUDNN
