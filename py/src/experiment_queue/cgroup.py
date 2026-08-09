from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CgroupV2MemoryScope:
    directory: Path

    def validate(self) -> None:
        if not self.directory.is_dir():
            raise ValueError(f'Cgroup directory does not exist: {self.directory}')
        required_files = (
            self._events_path,
            self._kill_path,
            self._memory_max_path,
            self._memory_oom_group_path,
            self._memory_swap_max_path,
            self._processes_path,
        )
        missing_files = tuple(path for path in required_files if not path.is_file())
        if missing_files:
            raise ValueError(f'Cgroup v2 memory scope is missing required files: {missing_files}')
        unwritable_files = tuple(
            path
            for path in (
                self._kill_path,
                self._memory_max_path,
                self._memory_oom_group_path,
                self._memory_swap_max_path,
                self._processes_path,
            )
            if not os.access(path, os.W_OK)
        )
        if unwritable_files:
            raise ValueError(f'Cgroup v2 memory scope files are not writable: {unwritable_files}')
        if self.populated:
            raise ValueError(f'Cgroup v2 memory scope is already populated: {self.directory}')

    def prepare(self, memory_limit_bytes: int) -> None:
        if memory_limit_bytes <= 0:
            raise ValueError('Cgroup memory limit must be positive.')
        self.validate()
        self._memory_max_path.write_text(f'{memory_limit_bytes}\n', encoding='ascii')
        self._memory_swap_max_path.write_text('0\n', encoding='ascii')
        self._memory_oom_group_path.write_text('1\n', encoding='ascii')

    def validate_process_migration(self) -> None:
        probe_path = Path(__file__).with_name('cgroup_probe.py').resolve()
        probe = subprocess.run(
            (sys.executable, str(probe_path), '--cgroup-processes', str(self._processes_path)),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            message = probe.stderr.strip() or f'probe exited with code {probe.returncode}'
            raise ValueError(f'Cannot move a queue child into cgroup {self.directory}: {message}')
        if self.populated:
            raise ValueError(f'Cgroup validation probe did not leave the scope: {self.directory}')

    @property
    def populated(self) -> bool:
        for line in self._events_path.read_text(encoding='ascii').splitlines():
            name, value = line.split(maxsplit=1)
            if name == 'populated':
                if value not in {'0', '1'}:
                    raise ValueError(f'Invalid cgroup populated value: {value!r}')
                return value == '1'
        raise ValueError(f'Cgroup events do not report populated state: {self._events_path}')

    def kill(self) -> None:
        self._kill_path.write_text('1\n', encoding='ascii')

    @property
    def processes_path(self) -> Path:
        return self._processes_path

    @property
    def _events_path(self) -> Path:
        return self.directory / 'cgroup.events'

    @property
    def _kill_path(self) -> Path:
        return self.directory / 'cgroup.kill'

    @property
    def _memory_max_path(self) -> Path:
        return self.directory / 'memory.max'

    @property
    def _memory_oom_group_path(self) -> Path:
        return self.directory / 'memory.oom.group'

    @property
    def _memory_swap_max_path(self) -> Path:
        return self.directory / 'memory.swap.max'

    @property
    def _processes_path(self) -> Path:
        return self.directory / 'cgroup.procs'
