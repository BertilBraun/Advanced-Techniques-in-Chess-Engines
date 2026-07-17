import random
from pathlib import Path

import pytest

from src.experiment.run_configuration import load_run_configuration
from tools import prepare_chess_evaluation_dataset


def test_prepare_dataset_records_source_and_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configuration = load_run_configuration(Path('configs/chess-clean-4x4070-main.json'))
    dataset_path = tmp_path / 'memory.hdf5'
    manifest_path = tmp_path / 'manifest.json'

    def create_dataset(requested_path: str) -> None:
        assert Path(requested_path) == dataset_path
        Path(requested_path).write_bytes(str(random.random()).encode('ascii'))

    monkeypatch.setattr(prepare_chess_evaluation_dataset, 'SOURCE_ROOT', tmp_path)
    monkeypatch.setattr(prepare_chess_evaluation_dataset, 'ensure_eval_dataset_exists', create_dataset)
    configuration = configuration.model_copy(
        update={
            'evaluation_protocol': configuration.evaluation_protocol.model_copy(
                update={'evaluation_dataset_path': 'memory.hdf5'}
            )
        }
    )

    manifest = prepare_chess_evaluation_dataset.prepare_dataset(configuration, manifest_path)

    assert manifest.dataset_path == 'memory.hdf5'
    assert manifest.source_game_count == 50
    assert manifest.random_seed == configuration.workload.random_seed
    assert manifest.dataset_sha256 == prepare_chess_evaluation_dataset.file_sha256(dataset_path)
    assert manifest_path.is_file()
