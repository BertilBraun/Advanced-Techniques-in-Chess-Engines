from __future__ import annotations

from typing import TYPE_CHECKING

from src.self_play.configuration import (
    BatchedInferenceParams,
    InferenceMemoryFormat,
    InferencePrecision,
    SdpaBackend,
)

if TYPE_CHECKING:
    from AlphaZeroCpp import InferenceExecutionOptions as NativeInferenceExecutionOptions
    from AlphaZeroCpp import InferenceMemoryFormat as NativeInferenceMemoryFormat
    from AlphaZeroCpp import InferencePrecision as NativeInferencePrecision
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


def native_inference_precision(precision: InferencePrecision) -> NativeInferencePrecision:
    from AlphaZeroCpp import InferencePrecision as NativeInferencePrecision

    match precision:
        case InferencePrecision.BFLOAT16:
            return NativeInferencePrecision.BFLOAT16
        case InferencePrecision.FLOAT16:
            return NativeInferencePrecision.FLOAT16
        case InferencePrecision.FLOAT32:
            return NativeInferencePrecision.FLOAT32


def native_inference_memory_format(memory_format: InferenceMemoryFormat) -> NativeInferenceMemoryFormat:
    from AlphaZeroCpp import InferenceMemoryFormat as NativeInferenceMemoryFormat

    match memory_format:
        case InferenceMemoryFormat.CONTIGUOUS:
            return NativeInferenceMemoryFormat.CONTIGUOUS
        case InferenceMemoryFormat.CHANNELS_LAST:
            return NativeInferenceMemoryFormat.CHANNELS_LAST


def native_execution_options(inference: BatchedInferenceParams) -> NativeInferenceExecutionOptions:
    from AlphaZeroCpp import InferenceExecutionOptions as NativeInferenceExecutionOptions

    return NativeInferenceExecutionOptions(
        sdpa_backend=native_sdpa_backend(inference.sdpa_backend),
        precision=native_inference_precision(inference.precision),
        memory_format=native_inference_memory_format(inference.memory_format),
        cudnn_benchmark=inference.cudnn_benchmark,
    )
