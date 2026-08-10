from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile


@dataclass(frozen=True)
class LauncherSnapshot:
    directory: Path
    linux_child: Path
    worktree_child: Path

    def close(self) -> None:
        if self.directory.parent.name != '.queue-launchers':
            raise ValueError(f'Refusing to remove unexpected launcher snapshot: {self.directory}')
        shutil.rmtree(self.directory)


def experiment_queue_source_directory() -> Path:
    return Path(__file__).resolve().parent


def create_launcher_snapshot(destination_root: Path, source_directory: Path) -> LauncherSnapshot:
    launcher_root = destination_root / '.queue-launchers'
    launcher_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='supervisor-', dir=launcher_root))
    linux_child = _copy_helper(source_directory / 'linux_child.py', directory)
    worktree_child = _copy_helper(source_directory / 'worktree_child.py', directory)
    return LauncherSnapshot(
        directory=directory,
        linux_child=linux_child,
        worktree_child=worktree_child,
    )


def _copy_helper(source_path: Path, destination_directory: Path) -> Path:
    if not source_path.is_file():
        raise ValueError(f'Queue launcher helper does not exist: {source_path}')
    destination_path = destination_directory / source_path.name
    destination_path.write_bytes(source_path.read_bytes())
    return destination_path
