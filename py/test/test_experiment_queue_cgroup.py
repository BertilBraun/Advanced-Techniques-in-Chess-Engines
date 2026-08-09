from pathlib import Path

import pytest

from src.experiment_queue.cgroup import CgroupV2MemoryScope


def _fake_cgroup(directory: Path) -> CgroupV2MemoryScope:
    directory.mkdir()
    for filename, content in (
        ('cgroup.events', 'populated 0\nfrozen 0\n'),
        ('cgroup.kill', ''),
        ('cgroup.procs', ''),
        ('memory.max', 'max\n'),
        ('memory.oom.group', '0\n'),
        ('memory.swap.max', 'max\n'),
    ):
        (directory / filename).write_text(content, encoding='ascii')
    return CgroupV2MemoryScope(directory)


def test_cgroup_preparation_sets_aggregate_memory_policy_and_accepts_child_migration(tmp_path: Path) -> None:
    memory_scope = _fake_cgroup(tmp_path / 'cgroup')

    memory_scope.prepare(123_456)
    memory_scope.validate_process_migration()

    assert (memory_scope.directory / 'memory.max').read_text(encoding='ascii') == '123456\n'
    assert (memory_scope.directory / 'memory.swap.max').read_text(encoding='ascii') == '0\n'
    assert (memory_scope.directory / 'memory.oom.group').read_text(encoding='ascii') == '1\n'


def test_cgroup_validation_rejects_a_populated_scope(tmp_path: Path) -> None:
    memory_scope = _fake_cgroup(tmp_path / 'cgroup')
    (memory_scope.directory / 'cgroup.events').write_text('populated 1\n', encoding='ascii')

    with pytest.raises(ValueError, match='already populated'):
        memory_scope.validate()


def test_cgroup_validation_rejects_missing_controller_files(tmp_path: Path) -> None:
    memory_scope = _fake_cgroup(tmp_path / 'cgroup')
    (memory_scope.directory / 'memory.max').unlink()

    with pytest.raises(ValueError, match='missing required files'):
        memory_scope.validate()
