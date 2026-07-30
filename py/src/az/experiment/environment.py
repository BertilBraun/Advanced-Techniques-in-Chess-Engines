from __future__ import annotations

import ctypes
import os
import platform
import shutil
from pathlib import Path

import torch

from src.az.config.manifest import HardwareDeclaration


class _WindowsMemoryStatus(ctypes.Structure):
    _fields_ = (
        ('length', ctypes.c_ulong),
        ('memory_load', ctypes.c_ulong),
        ('total_physical', ctypes.c_ulonglong),
        ('available_physical', ctypes.c_ulonglong),
        ('total_page_file', ctypes.c_ulonglong),
        ('available_page_file', ctypes.c_ulonglong),
        ('total_virtual', ctypes.c_ulonglong),
        ('available_virtual', ctypes.c_ulonglong),
        ('available_extended_virtual', ctypes.c_ulonglong),
    )


def inspect_hardware(output_directory: Path) -> HardwareDeclaration:
    logical_cpu_count = os.cpu_count()
    if logical_cpu_count is None:
        raise ValueError('Logical CPU count is unavailable.')
    gpu_count = torch.cuda.device_count()
    gpu_models = tuple(torch.cuda.get_device_name(index) for index in range(gpu_count))
    if len(set(gpu_models)) > 1:
        raise ValueError('The current hardware manifest requires homogeneous GPU models.')
    return HardwareDeclaration(
        gpu_model='none' if not gpu_models else gpu_models[0],
        gpu_count=gpu_count,
        logical_cpu_count=logical_cpu_count,
        ram_gib=_physical_memory_bytes() / 2**30,
        free_disk_gib=shutil.disk_usage(output_directory.parent).free / 2**30,
    )


def _physical_memory_bytes() -> int:
    match platform.system():
        case 'Windows':
            status = _WindowsMemoryStatus()
            status.length = ctypes.sizeof(status)
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError(ctypes.get_last_error(), 'GlobalMemoryStatusEx failed.')
            return int(status.total_physical)
        case 'Linux':
            return int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
        case system:
            raise ValueError(f'Physical-memory inspection is unsupported on {system}.')
