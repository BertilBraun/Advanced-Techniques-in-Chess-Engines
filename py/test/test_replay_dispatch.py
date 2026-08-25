from __future__ import annotations

import os
from pathlib import Path

import pytest
from src.replay.dispatch import (
    InboxDispatcher,
    next_worker_counter,
    parse_worker_source_file_name,
    worker_source_file_name,
    worker_source_file_names,
)

GAME_NAME = 'worker-3-process-38c8809f-a49d-4d98-8da5-034614893665-game-00000000000000000007.json'


def _inbox_and_workers(tmp_path: Path, worker_count: int) -> tuple[Path, tuple[Path, ...]]:
    inbox = tmp_path / 'inbox'
    inbox.mkdir()
    workers = tuple(tmp_path / f'worker-{index}' for index in range(worker_count))
    for worker in workers:
        worker.mkdir()
    return inbox, workers


def test_worker_source_names_round_trip_through_their_counter_prefix() -> None:
    name = worker_source_file_name(42, GAME_NAME)

    assert name == f'000000000000042-{GAME_NAME}'
    assert parse_worker_source_file_name(name) == (42, GAME_NAME)


@pytest.mark.parametrize('file_name', ('no-prefix.json', '42-short.json', 'abcdefghijklmno-x.json', 'nodash'))
def test_unprefixed_worker_source_names_are_rejected(file_name: str) -> None:
    with pytest.raises(ValueError, match='invalid'):
        parse_worker_source_file_name(file_name)


def test_counters_are_reseeded_from_the_highest_prefix_already_in_the_directory(tmp_path: Path) -> None:
    inbox, workers = _inbox_and_workers(tmp_path, 1)
    (workers[0] / worker_source_file_name(7, GAME_NAME)).write_text('{}', encoding='utf-8')
    assert next_worker_counter(workers[0]) == 8
    (inbox / GAME_NAME).write_text('{}', encoding='utf-8')

    InboxDispatcher(inbox, workers).dispatch_once()

    assert sorted(worker_source_file_names(workers[0])) == [
        worker_source_file_name(7, GAME_NAME),
        worker_source_file_name(8, GAME_NAME),
    ]


def test_a_dispatch_pass_reads_no_more_than_its_rename_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inbox, workers = _inbox_and_workers(tmp_path, 2)
    for index in range(50):
        (inbox / f'worker-3-process-38c8809f-a49d-4d98-8da5-034614893665-game-{index:020d}.json').write_text(
            '{}', encoding='utf-8'
        )
    renames = 0
    original_rename = os.rename

    def counted_rename(source: object, target: object) -> None:
        nonlocal renames
        renames += 1
        original_rename(source, target)  # type: ignore[arg-type]

    monkeypatch.setattr(os, 'rename', counted_rename)
    dispatcher = InboxDispatcher(inbox, workers, rename_cap=4)

    assert dispatcher.dispatch_once() == 4
    assert renames == 4
    assert len(tuple(inbox.glob('*.json'))) == 46


def test_non_game_inbox_entries_are_left_alone(tmp_path: Path) -> None:
    inbox, workers = _inbox_and_workers(tmp_path, 1)
    (inbox / '.partial.tmp').write_text('partial', encoding='utf-8')
    (inbox / GAME_NAME).write_text('{}', encoding='utf-8')

    assert InboxDispatcher(inbox, workers).dispatch_once() == 1
    assert [path.name for path in inbox.iterdir()] == ['.partial.tmp']


def _write_inbox_games(inbox: Path, count: int) -> None:
    for index in range(count):
        (inbox / f'worker-0-process-abc-game-{index:020d}.json').write_text('{}', encoding='utf-8')


def test_dispatch_levels_worker_depth_instead_of_handing_out_equal_shares(tmp_path: Path) -> None:
    inbox, workers = _inbox_and_workers(tmp_path, 4)
    for backlog_index in range(9):
        name = worker_source_file_name(backlog_index, f'worker-0-process-abc-game-{backlog_index:020d}.json')
        (workers[0] / name).write_text('{}', encoding='utf-8')
    _write_inbox_games(inbox, 12)

    dispatcher = InboxDispatcher(inbox, workers)
    assert dispatcher.dispatch_once() == 12

    # A blind round robin would have handed the already-deep worker three more and left it at 12.
    assert [len(worker_source_file_names(worker)) for worker in workers] == [9, 4, 4, 4]


def test_dispatch_spreads_evenly_when_every_worker_starts_empty(tmp_path: Path) -> None:
    inbox, workers = _inbox_and_workers(tmp_path, 4)
    _write_inbox_games(inbox, 8)

    assert InboxDispatcher(inbox, workers).dispatch_once() == 8

    assert [len(worker_source_file_names(worker)) for worker in workers] == [2, 2, 2, 2]
