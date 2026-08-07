import random
from pathlib import Path

import pytest

from src.experiment.configuration import load_chess_experiment_configuration
from tools import prepare_chess_evaluation_dataset


def test_prepare_dataset_records_source_and_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configuration = load_chess_experiment_configuration(Path('configs/chess-experiment-template.yaml'))
    dataset_path = tmp_path / 'memory.hdf5'
    manifest_path = tmp_path / 'manifest.json'

    def create_dataset(requested_path: Path) -> None:
        assert requested_path == dataset_path
        requested_path.write_bytes(str(random.random()).encode('ascii'))

    monkeypatch.setattr(prepare_chess_evaluation_dataset, 'SOURCE_ROOT', tmp_path)
    monkeypatch.setattr(prepare_chess_evaluation_dataset, 'ensure_evaluation_dataset_exists', create_dataset)
    evaluation = configuration.chess.evaluation.validated_copy(update={'dataset_path': 'memory.hdf5'})
    chess = configuration.chess.validated_copy(update={'evaluation': evaluation.model_dump(mode='json')})
    configuration = configuration.validated_copy(update={'chess': chess.model_dump(mode='json')})

    manifest = prepare_chess_evaluation_dataset.prepare_dataset(configuration, manifest_path)

    assert manifest.dataset_path == 'memory.hdf5'
    assert manifest.source_game_count == 50
    assert manifest.random_seed == configuration.training.random_seed
    assert manifest.dataset_sha256 == prepare_chess_evaluation_dataset.file_sha256(dataset_path)
    assert manifest_path.is_file()
