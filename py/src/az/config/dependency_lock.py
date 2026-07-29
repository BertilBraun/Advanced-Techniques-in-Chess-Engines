from __future__ import annotations

import re
from pathlib import Path

from pydantic import Field

from src.az.config.base import FrozenModel


_PINNED_REQUIREMENT = re.compile(r'^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;\\]+)\s*\\?$')


class DependencyRecord(FrozenModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


def normalized_package_name(name: str) -> str:
    return name.casefold().replace('_', '-').replace('.', '-')


def parse_pinned_dependency_lock(path: Path) -> tuple[DependencyRecord, ...]:
    if not path.is_file():
        raise ValueError(f'Dependency lock file does not exist: {path}')
    records: list[DependencyRecord] = []
    for line_number, raw_line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith('#') or line.startswith('--hash='):
            continue
        if line.startswith(('-r ', '--requirement ', '-c ', '--constraint ')):
            raise ValueError(f'Recursive requirement and constraint files are unsupported: {path}:{line_number}')
        if line.startswith('-'):
            raise ValueError(f'Unsupported dependency lock option at {path}:{line_number}: {line}')
        match = _PINNED_REQUIREMENT.fullmatch(line)
        if match is None:
            raise ValueError(f'Dependency must use an exact name==version pin at {path}:{line_number}: {line}')
        records.append(
            DependencyRecord(
                name=match.group('name'),
                version=match.group('version'),
            )
        )
    names = tuple(normalized_package_name(record.name) for record in records)
    if len(set(names)) != len(names):
        raise ValueError(f'Dependency lock contains duplicate normalized package names: {path}')
    return tuple(sorted(records, key=lambda record: normalized_package_name(record.name)))
