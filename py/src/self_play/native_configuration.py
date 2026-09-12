from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from src.self_play.configuration import (
    BatchedInferenceParams,
    InferenceBackendConfiguration,
    InferenceMemoryFormat,
    InferencePrecision,
    SdpaBackend,
    TensorRtInferenceBackend,
    TorchScriptInferenceBackend,
)

if TYPE_CHECKING:
    from AlphaZeroCpp import InferenceBackend as NativeInferenceBackend
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


def native_inference_backend(backend: InferenceBackendConfiguration) -> NativeInferenceBackend:
    from AlphaZeroCpp import InferenceBackend as NativeInferenceBackend

    match backend:
        case TorchScriptInferenceBackend():
            return NativeInferenceBackend.TORCHSCRIPT
        case TensorRtInferenceBackend():
            return NativeInferenceBackend.TENSORRT


def resolved_inference_model_path(model_path: Path, backend: InferenceBackendConfiguration) -> Path:
    match backend:
        case TorchScriptInferenceBackend():
            return model_path
        case TensorRtInferenceBackend(template_engine_path=template_engine_path):
            if model_path.name.endswith('.engine'):
                return model_path
            publisher = Path(__file__).parents[2] / 'tools' / 'publish_tensorrt_engine.py'
            completed = subprocess.run(
                (
                    sys.executable,
                    str(publisher),
                    '--model',
                    str(model_path),
                    '--template-engine',
                    str(template_engine_path),
                ),
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            return Path(payload['engine_path'])


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
