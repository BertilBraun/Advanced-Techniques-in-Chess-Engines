from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepositoryFile:
    relative_path: str
    source_path: Path


@dataclass(frozen=True)
class GitRepository:
    directory: Path
    revision: str


def run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ('git', '-C', str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def create_repository(directory: Path, files: tuple[RepositoryFile, ...]) -> GitRepository:
    directory.mkdir(parents=True, exist_ok=True)
    resolved_directory = directory.resolve()
    run_git(resolved_directory, 'init')
    run_git(resolved_directory, 'config', 'user.name', 'Queue Test')
    run_git(resolved_directory, 'config', 'user.email', 'queue-test@example.com')
    for repository_file in files:
        destination = resolved_directory / repository_file.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository_file.source_path, destination)
    return GitRepository(directory=resolved_directory, revision=commit_all(resolved_directory, 'initial'))


def commit_all(directory: Path, message: str) -> str:
    run_git(directory, 'add', '--all')
    run_git(directory, 'commit', '--message', message)
    return run_git(directory, 'rev-parse', 'HEAD')
