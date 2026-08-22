from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SourceRevision:
    commit: str
    dirty: bool


def read_source_revision() -> SourceRevision:
    revision = read_source_revision_if_available()
    if revision is None:
        raise RuntimeError(f'git could not report the source revision of {REPOSITORY_ROOT}.')
    return revision


def read_source_revision_if_available() -> SourceRevision | None:
    commit = _git_output(('rev-parse', 'HEAD'))
    status = _git_output(('status', '--porcelain'))
    if commit is None or status is None:
        return None
    return SourceRevision(commit=commit, dirty=bool(status))


def _git_output(arguments: tuple[str, ...]) -> str | None:
    try:
        completed = subprocess.run(
            ('git', *arguments),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()
