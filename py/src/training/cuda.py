import os
from typing import Protocol


class StartableProcess(Protocol):
    def start(self) -> None: ...


def start_process_on_cuda_device(process: StartableProcess, physical_device_id: int) -> None:
    variable_name = 'CUDA_VISIBLE_DEVICES'
    previous_value = os.environ.get(variable_name)
    os.environ[variable_name] = str(physical_device_id)
    try:
        process.start()
    finally:
        if previous_value is None:
            del os.environ[variable_name]
        else:
            os.environ[variable_name] = previous_value
