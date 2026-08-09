from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import psutil

from src.experiment_queue.scheduler import ResourceAssignment


@dataclass
class RunningProcess:
    experiment_id: str
    assignment: ResourceAssignment
    process: subprocess.Popen[bytes]
    stdout_stream: BinaryIO
    stderr_stream: BinaryIO
    tracked_processes: dict[int, psutil.Process]

    def close_logs(self) -> None:
        self.stdout_stream.close()
        self.stderr_stream.close()

    def refresh_process_tree(self) -> None:
        for process in tuple(self.tracked_processes.values()):
            try:
                descendants = process.children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            for descendant in descendants:
                self.tracked_processes.setdefault(descendant.pid, descendant)

    @property
    def resident_memory_bytes(self) -> int:
        self.refresh_process_tree()
        resident_memory_bytes = 0
        for process in self.tracked_processes.values():
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    resident_memory_bytes += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return resident_memory_bytes

    @property
    def has_live_processes(self) -> bool:
        self.refresh_process_tree()
        for process in self.tracked_processes.values():
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False


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
        tracked_processes={process.pid: psutil.Process(process.pid)},
    )


def terminate_process_group(running_process: RunningProcess, grace_seconds: float) -> int:
    running_process.refresh_process_tree()
    try:
        os.killpg(running_process.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    deadline = time.monotonic() + grace_seconds
    while running_process.has_live_processes and time.monotonic() < deadline:
        time.sleep(0.01)
    if running_process.has_live_processes:
        for process in running_process.tracked_processes.values():
            try:
                process.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

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
