from __future__ import annotations

import json
import subprocess
import sys
import time
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
from src.util.log import log

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


def _uses_torchscript_bootstrap(model_generation: int, backend: TensorRtInferenceBackend) -> bool:
    return backend.bootstrap_with_torchscript and model_generation == 0


def native_inference_backend(
    backend: InferenceBackendConfiguration,
    model_generation: int,
) -> NativeInferenceBackend:
    from AlphaZeroCpp import InferenceBackend as NativeInferenceBackend

    match backend:
        case TorchScriptInferenceBackend():
            return NativeInferenceBackend.TORCHSCRIPT
        case TensorRtInferenceBackend() as tensor_rt_backend if _uses_torchscript_bootstrap(
            model_generation,
            tensor_rt_backend,
        ):
            return NativeInferenceBackend.TORCHSCRIPT
        case TensorRtInferenceBackend():
            return NativeInferenceBackend.TENSORRT


def resolved_inference_model_path(
    model_path: Path,
    backend: InferenceBackendConfiguration,
    model_generation: int,
) -> Path:
    match backend:
        case TorchScriptInferenceBackend():
            return model_path
        case TensorRtInferenceBackend() as tensor_rt_backend if _uses_torchscript_bootstrap(
            model_generation,
            tensor_rt_backend,
        ):
            if not model_path.name.endswith('.jit.pt'):
                raise ValueError('The generation-0 TensorRT bootstrap artifact must be TorchScript.')
            log(f'Using TorchScript bootstrap inference artifact {model_path}.')
            return model_path
        case TensorRtInferenceBackend(template_engine_paths=template_engine_paths):
            started_at = time.perf_counter()
            if model_path.name.endswith('.engine'):
                return model_path
            publisher = Path(__file__).parents[2] / 'tools' / 'publish_tensorrt_engine.py'
            template_arguments = tuple(
                argument
                for template_engine_path in template_engine_paths
                for argument in ('--template-engine', str(template_engine_path))
            )
            completed = subprocess.run(
                (
                    sys.executable,
                    '-m',
                    'tools.publish_tensorrt_engine',
                    '--model',
                    str(model_path.resolve()),
                    *template_arguments,
                ),
                check=True,
                capture_output=True,
                text=True,
                cwd=publisher.parent.parent,
            )
            payload = json.loads(completed.stdout)
            log(
                f'Published TensorRT inference artifact for {model_path.name} in '
                f'{time.perf_counter() - started_at:.3f}s.'
            )
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
