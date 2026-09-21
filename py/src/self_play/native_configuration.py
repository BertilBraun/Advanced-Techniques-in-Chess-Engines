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
    TensorRtTemplatePrecision,
    TorchScriptInferenceBackend,
)
from src.training.quantization.configuration import QatCheckpointPhase
from src.util.log import log

if TYPE_CHECKING:
    from AlphaZeroCpp import InferenceBackend as NativeInferenceBackend
    from AlphaZeroCpp import InferenceExecutionOptions as NativeInferenceExecutionOptions
    from AlphaZeroCpp import InferenceMemoryFormat as NativeInferenceMemoryFormat
    from AlphaZeroCpp import InferencePrecision as NativeInferencePrecision
    from AlphaZeroCpp import SdpaBackend as NativeSdpaBackend


FIDELITY_PROBE_FILE_NAME = 'fidelity-probe.npz'


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


def uses_torchscript_bootstrap(model_generation: int, backend: InferenceBackendConfiguration) -> bool:
    match backend:
        case TensorRtInferenceBackend(bootstrap_with_torchscript=True):
            return model_generation == 0
        case _:
            return False


def native_inference_backend(
    backend: InferenceBackendConfiguration,
    model_generation: int,
) -> NativeInferenceBackend:
    from AlphaZeroCpp import InferenceBackend as NativeInferenceBackend

    match backend:
        case TorchScriptInferenceBackend():
            return NativeInferenceBackend.TORCHSCRIPT
        case TensorRtInferenceBackend() as tensor_rt_backend if uses_torchscript_bootstrap(
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
    model_id: str,
    qat_phase: QatCheckpointPhase | None,
) -> Path:
    match backend:
        case TorchScriptInferenceBackend():
            return model_path
        case TensorRtInferenceBackend() as tensor_rt_backend if uses_torchscript_bootstrap(
            model_generation,
            tensor_rt_backend,
        ):
            if not model_path.name.endswith('.jit.pt'):
                raise ValueError('The generation-0 TensorRT bootstrap artifact must be TorchScript.')
            log(f'Using TorchScript bootstrap inference artifact {model_path}.')
            return model_path
        case TensorRtInferenceBackend() as tensor_rt_backend:
            started_at = time.perf_counter()
            if model_path.name.endswith('.engine'):
                return model_path
            template_precision = (
                TensorRtTemplatePrecision.FLOAT
                if model_path.name.endswith('.fp16.onnx')
                else TensorRtTemplatePrecision.INT8
            )
            template_engine_path = tensor_rt_backend.template_engine_path(
                model_id,
                template_precision,
                qat_phase,
            )
            publisher = Path(__file__).parents[2] / 'tools' / 'publish_tensorrt_engine.py'
            command = (
                sys.executable,
                '-m',
                'tools.publish_tensorrt_engine',
                '--model',
                str(model_path.resolve()),
                '--template-engine',
                str(template_engine_path),
            )
            if tensor_rt_backend.allow_fidelity_deviation:
                command += ('--allow-fidelity-deviation',)
            probe_path = model_path.parent / FIDELITY_PROBE_FILE_NAME
            if probe_path.is_file():
                command += ('--probe-states', str(probe_path.resolve()))
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=publisher.parent.parent,
            )
            payload = json.loads(completed.stdout.splitlines()[-1])
            log(
                f'Published TensorRT inference artifact for {model_path.name} in '
                f'{time.perf_counter() - started_at:.3f}s.'
            )
            if not payload.get('fidelity_limits_passed', True):
                log(
                    f'TensorRT fidelity warning for {model_path.name}: '
                    f'policy top1/KL mean/max={payload["policy_top1_agreement"]:.6f}/'
                    f'{payload["policy_mean_kl_divergence"]:.6f}/'
                    f'{payload["policy_maximum_kl_divergence"]:.6f}, '
                    f'WDL mean/max={payload["wdl_mean_absolute_error"]:.6f}/'
                    f'{payload["wdl_maximum_absolute_error"]:.6f}.'
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
