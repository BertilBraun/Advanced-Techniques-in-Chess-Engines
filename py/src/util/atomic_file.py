from __future__ import annotations

import os
import uuid
from pathlib import Path


def write_text_atomically(path: Path, content: str, *, sync_directory: bool = True) -> None:
    write_bytes_atomically(path, content.encode('utf-8'), sync_directory=sync_directory)


def write_bytes_atomically(path: Path, content: bytes, *, sync_directory: bool = True) -> None:
    # A directory fsync costs the same as the file fsync on this filesystem, so a caller writing a
    # batch may defer it and sync the directory once for the whole batch.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary_path.open('xb') as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        if sync_directory:
            fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def fsync_directory(path: Path) -> None:
    if os.name == 'nt':
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
