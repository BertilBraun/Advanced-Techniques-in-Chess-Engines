from __future__ import annotations

from pathlib import Path

from src.games.chess import database


def ensure_evaluation_dataset_exists(dataset_path: Path) -> None:
    if dataset_path.exists():
        return
    output_paths = database.process_month(2024, 10, num_games_per_month=50)
    assert len(output_paths) == 1
    output_paths[0].rename(dataset_path)
