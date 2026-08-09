from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from src.experiment_queue.cgroup import CgroupV2MemoryScope
from src.experiment_queue.scheduler import ResourceAssignment


@dataclass
class RunningProcess:
    experiment_id: str
    assignment: ResourceAssignment
    process: subprocess.Popen[bytes]
    stdout_stream: BinaryIO
    stderr_stream: BinaryIO
    memory_scope: CgroupV2MemoryScope

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
    memory_scope = CgroupV2MemoryScope(assignment.cgroup_directory)
    memory_scope.prepare(assignment.ram_limit_bytes)
    wrapper_command = (
        sys.executable,
        str(child_wrapper),
        '--cpu-affinity',
        ','.join(str(cpu_index) for cpu_index in assignment.cpu_affinity),
        '--cgroup-processes',
        str(memory_scope.processes_path),
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
        memory_scope=memory_scope,
    )


def terminate_process_group(running_process: RunningProcess, grace_seconds: float) -> int:
    try:
        os.killpg(running_process.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    deadline = time.monotonic() + grace_seconds
    while running_process.memory_scope.populated and time.monotonic() < deadline:
        time.sleep(0.01)
    if running_process.memory_scope.populated:
        running_process.memory_scope.kill()
        kill_deadline = time.monotonic() + max(1.0, grace_seconds)
        while running_process.memory_scope.populated and time.monotonic() < kill_deadline:
            time.sleep(0.01)
        if running_process.memory_scope.populated:
            raise RuntimeError(f'Cgroup remained populated after cgroup.kill: {running_process.memory_scope.directory}')

    return_code = running_process.process.poll()
    if return_code is not None:
        return return_code
    remaining_grace_seconds = max(1.0, deadline - time.monotonic())
    try:
        return running_process.process.wait(timeout=remaining_grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(running_process.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return running_process.process.wait()
