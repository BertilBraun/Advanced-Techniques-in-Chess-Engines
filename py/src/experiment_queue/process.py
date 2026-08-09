from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from src.experiment_queue.scheduler import ResourceAssignment


@dataclass
class RunningProcess:
    experiment_id: str
    assignment: ResourceAssignment
    process: subprocess.Popen[bytes]
    stdout_stream: BinaryIO
    stderr_stream: BinaryIO

    def close_logs(self) -> None:
        self.stdout_stream.close()
        self.stderr_stream.close()


def launch_process(
    experiment_id: str,
    command: tuple[str, ...],
    assignment: ResourceAssignment,
    stdout_path: Path,
    stderr_path: Path,
) -> RunningProcess:
    if sys.platform != 'linux':
        raise ValueError('The experiment queue launcher supports Linux only.')

    child_wrapper = Path(__file__).with_name('linux_child.py').resolve()
    wrapper_command = (
        sys.executable,
        str(child_wrapper),
        '--cpu-affinity',
        ','.join(str(cpu_index) for cpu_index in assignment.cpu_affinity),
        '--ram-limit-bytes',
        str(assignment.ram_limit_bytes),
        *command,
    )
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = ','.join(str(device) for device in assignment.cuda_devices)
    stdout_stream = stdout_path.open('xb')
    try:
        stderr_stream = stderr_path.open('xb')
    except Exception:
        stdout_stream.close()
        raise
    try:
        process = subprocess.Popen(
            wrapper_command,
            cwd=assignment.working_directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_stream,
            stderr=stderr_stream,
            start_new_session=True,
        )
    except Exception:
        stdout_stream.close()
        stderr_stream.close()
        raise
    return RunningProcess(
        experiment_id=experiment_id,
        assignment=assignment,
        process=process,
        stdout_stream=stdout_stream,
        stderr_stream=stderr_stream,
    )


def terminate_process_group(running_process: RunningProcess, grace_seconds: float) -> int:
    return_code = running_process.process.poll()
    if return_code is not None:
        return return_code

    try:
        os.killpg(running_process.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return running_process.process.wait()
    try:
        return running_process.process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(running_process.process.pid, signal.SIGKILL)
        return running_process.process.wait()
