from __future__ import annotations

from src.az.self_play.scheduling import LogicalWorkerGameScheduler


def test_logical_worker_game_lineage_is_unique_and_stable_across_process_ranges() -> None:
    first_process = LogicalWorkerGameScheduler(first_worker_index=0, worker_count=2)
    second_process = LogicalWorkerGameScheduler(first_worker_index=2, worker_count=2)

    first_coordinates = tuple(first_process.next_game() for _ in range(6))
    second_coordinates = tuple(second_process.next_game() for _ in range(4))

    assert tuple((game.logical_worker_index, game.game_index) for game in first_coordinates) == (
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
    )
    assert tuple((game.logical_worker_index, game.game_index) for game in second_coordinates) == (
        (2, 0),
        (3, 0),
        (2, 1),
        (3, 1),
    )
    assert set(first_coordinates).isdisjoint(second_coordinates)
