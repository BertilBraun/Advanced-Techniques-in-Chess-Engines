from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class ChildCommand:
    name: str
    command: tuple[str, ...]
    log_path: Path


@dataclass(frozen=True)
class ChildOutcome:
    name: str
    command: tuple[str, ...]
    log_path: Path
    return_code: int
    duration_seconds: float


@dataclass
class _RunningChild:
    specification: ChildCommand
    process: subprocess.Popen[str]
    log_handle: TextIO
    started_at: float


def parse_devices(values: tuple[int, ...], label: str) -> tuple[int, ...]:
    if not values or any(device < 0 for device in values) or len(set(values)) != len(values):
        raise ValueError(f'{label} devices must be nonempty, unique, and nonnegative.')
    return values


def validate_input_paths(paths: tuple[Path, ...]) -> None:
    missing = tuple(path for path in paths if not path.exists())
    if missing:
        raise ValueError(f'Required input paths do not exist: {missing}')


def validate_new_output_root(output_root: Path) -> None:
    if output_root.exists():
        raise ValueError(f'Output root already exists: {output_root}')


def run_child_commands(
    specifications: tuple[ChildCommand, ...],
    dry_run: bool,
    poll_interval_seconds: float = 1.0,
) -> tuple[ChildOutcome, ...]:
    if not specifications:
        raise ValueError('At least one child command is required.')
    if dry_run:
        for specification in specifications:
            print(f'[{specification.name}] {subprocess.list2cmdline(specification.command)}', flush=True)
        return ()

    running: list[_RunningChild] = []
    outcomes: list[ChildOutcome] = []
    try:
        for specification in specifications:
            specification.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = specification.log_path.open('w', encoding='utf-8')
            process = subprocess.Popen(
                specification.command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append(_RunningChild(specification, process, log_handle, time.monotonic()))
            print(f'started {specification.name}: PID {process.pid}', flush=True)

        while running:
            for child in tuple(running):
                return_code = child.process.poll()
                if return_code is None:
                    continue
                child.log_handle.close()
                running.remove(child)
                outcome = ChildOutcome(
                    name=child.specification.name,
                    command=child.specification.command,
                    log_path=child.specification.log_path,
                    return_code=return_code,
                    duration_seconds=time.monotonic() - child.started_at,
                )
                outcomes.append(outcome)
                status = 'finished' if return_code == 0 else f'FAILED ({return_code})'
                print(f'{status} {outcome.name} after {outcome.duration_seconds / 60.0:.1f} min', flush=True)
            if running:
                time.sleep(poll_interval_seconds)
    except BaseException:
        for child in running:
            child.process.terminate()
        for child in running:
            child.process.wait()
            child.log_handle.close()
        raise
    return tuple(sorted(outcomes, key=lambda outcome: outcome.name))
