from __future__ import annotations

import subprocess
from pathlib import Path

import tensorrt
import torch
from src.training.quantization.runtime import specialize_qat_onnx_batch
from tools.build_tensorrt_refit_template import RefitMode, build_template
from tools.publish_tensorrt_engine import export_onnx, refit_engine, verify_engine

from deployment.web.backend.tensorrt_cache import TensorRtRuntimeIdentity


def modal_runtime_identity() -> TensorRtRuntimeIdentity:
    major, minor = torch.cuda.get_device_capability(0)
    cuda_runtime_version = torch.version.cuda
    if cuda_runtime_version is None:
        raise RuntimeError('The TensorRT deployment cannot identify the CUDA runtime version.')
    completed = subprocess.run(
        ('nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'),
        check=True,
        capture_output=True,
        text=True,
    )
    driver_versions = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if len(driver_versions) != 1:
        raise RuntimeError('The TensorRT deployment must expose exactly one NVIDIA driver version.')
    return TensorRtRuntimeIdentity(
        tensorrt_version=tensorrt.__version__,
        cuda_runtime_version=cuda_runtime_version,
        nvidia_driver_version=driver_versions[0],
        gpu_name=torch.cuda.get_device_name(0),
        gpu_compute_capability=f'{major}.{minor}',
    )


def build_and_verify_tensorrt_engine(
    source_path: Path,
    engine_path: Path,
    batch_size: int,
    channels: int,
    rows: int,
    columns: int,
    builder_optimization_level: int,
) -> None:
    onnx_path = engine_path.with_suffix('.onnx')
    template_path = engine_path.with_suffix('.template.engine')
    onnx_path.unlink(missing_ok=True)
    template_path.unlink(missing_ok=True)
    try:
        if source_path.suffix == '.onnx':
            specialize_qat_onnx_batch(source_path, onnx_path, batch_size)
        else:
            export_onnx(source_path, onnx_path, (batch_size, channels, rows, columns))
        build_template(
            onnx_path,
            template_path,
            batch_size,
            channels,
            rows,
            columns,
            builder_optimization_level,
            None,
            RefitMode.ALL,
        )
        refit_engine(template_path, onnx_path, engine_path)
        verify_engine(
            onnx_path,
            engine_path,
            (batch_size, channels, rows, columns),
            allow_fidelity_deviation=False,
        )
    finally:
        onnx_path.unlink(missing_ok=True)
        template_path.unlink(missing_ok=True)
