from __future__ import annotations

import subprocess

import tensorrt
import torch

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
