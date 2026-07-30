from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from src.az.training import checkpoints
from src.az.config.runtime import RetentionConfiguration
from src.az.experiment.artifact_retention import apply_checkpoint_retention
from src.az.replay.credits import EMPTY_CREDIT_PREFIX_SHA256, ReplayCreditState
from src.az.replay.sampling import ReplaySamplerState
from src.az.training.checkpoints import (
    CheckpointPointer,
    CheckpointPurpose,
    CheckpointRepository,
    TrainerCheckpointState,
)
from src.az.training.optimizer import LearningRateState


CONFIGURATION_SHA256 = 'a' * 64


def _state(model_version: int) -> TrainerCheckpointState:
    steps = model_version * 2
    earned = Decimal(100)
    consumed = Decimal(steps)
    return TrainerCheckpointState(
        replay_credits=ReplayCreditState(
            credited_unique_positions=100,
            credit_journal_prefix_sha256=EMPTY_CREDIT_PREFIX_SHA256,
            earned_position_credits=earned,
            consumed_position_credits=consumed,
            available_position_credits=earned - consumed,
            completed_optimizer_steps=steps,
            completed_training_quanta=model_version,
            model_version=model_version,
        ),
        replay_sampler=ReplaySamplerState(next_optimizer_step=steps),
        learning_rate=LearningRateState(completed_optimizer_steps=steps, current_learning_rate=0.01),
    )


def _repository(path: Path) -> CheckpointRepository:
    return CheckpointRepository(path.resolve(), UUID(int=1), CONFIGURATION_SHA256)


def test_checkpoint_publication_round_trips_opaque_artifacts(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    published = repository.publish(
        _state(1),
        CheckpointPurpose.SCHEDULED,
        b'model',
        b'optimizer',
        b'random',
    )
    restarted = _repository(tmp_path).load_current()

    assert restarted == published
    assert restarted.manifest.state == _state(1)
    assert restarted.model_artifact == b'model'
    assert restarted.optimizer_artifact == b'optimizer'
    assert restarted.torch_random_state_artifact == b'random'


def test_torn_pointer_and_incomplete_orphan_do_not_replace_published_checkpoint(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    original = repository.publish(
        _state(1),
        CheckpointPurpose.CREDIT_COMMIT,
        b'model-1',
        b'optimizer-1',
        b'random-1',
    )
    orphan = tmp_path / '.checkpoint-0000000002-orphan.partial'
    orphan.mkdir()
    (orphan / 'model.pt').write_bytes(b'incomplete')

    assert repository.load_current() == original
    repository.pointer_path.write_bytes(b'{"schema_version":')
    with pytest.raises(ValueError, match='invalid or torn'):
        repository.load_current()


def test_checkpoint_load_rejects_tampered_opaque_artifact(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    published = repository.publish(
        _state(1),
        CheckpointPurpose.CREDIT_COMMIT,
        b'model-1',
        b'optimizer-1',
        b'random-1',
    )
    model_path = (
        tmp_path / (f'checkpoint-0000000001-{published.manifest.checkpoint_id.hex}') / published.manifest.model.filename
    )
    model_path.write_bytes(b'tampered')

    with pytest.raises(ValueError, match='artifact checksum mismatch'):
        repository.load_current()


def test_failed_pointer_commit_leaves_previous_credit_state_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    original = repository.publish(
        _state(1),
        CheckpointPurpose.CREDIT_COMMIT,
        b'model-1',
        b'optimizer-1',
        b'random-1',
    )

    def fail_pointer_commit(path: Path, contents: bytes) -> None:
        del path, contents
        raise OSError('injected pointer failure')

    monkeypatch.setattr(checkpoints, '_atomic_replace', fail_pointer_commit)
    with pytest.raises(OSError, match='injected pointer failure'):
        repository.publish(
            _state(2),
            CheckpointPurpose.SCHEDULED,
            b'model-2',
            b'optimizer-2',
            b'random-2',
        )

    assert repository.load_current() == original


def test_retention_never_deletes_current_checkpoint(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    published_directories: list[str] = []
    for version in range(1, 5):
        loaded = repository.publish(
            _state(version),
            CheckpointPurpose.SCHEDULED,
            f'model-{version}'.encode(),
            f'optimizer-{version}'.encode(),
            f'random-{version}'.encode(),
        )
        published_directories.append(loaded.manifest.checkpoint_id.hex)
    current_directory = repository.load_current().manifest.checkpoint_id.hex

    result = apply_checkpoint_retention(
        tmp_path,
        RetentionConfiguration(
            recent_checkpoint_count=1,
            milestone_every_optimizer_steps=4,
            retain_replay_shards=True,
            retain_search_traces=False,
            retain_raw_evaluation_games=False,
        ),
    )

    assert any(current_directory in name for name in result.retained_checkpoint_directories)
    assert repository.load_current().manifest.state == _state(4)
    assert len(result.deleted_checkpoint_directories) == 2
    assert len(published_directories) == 4


def test_retention_preflight_prevents_partial_deletion_on_corrupt_old_manifest(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    for version in range(1, 4):
        repository.publish(
            _state(version),
            CheckpointPurpose.CREDIT_COMMIT,
            f'model-{version}'.encode(),
            f'optimizer-{version}'.encode(),
            f'random-{version}'.encode(),
        )
    directories = tuple(sorted(path for path in tmp_path.glob('checkpoint-*') if path.is_dir()))
    directories[1].joinpath('manifest.json').write_bytes(b'corrupt')
    before = tuple(path.name for path in directories)

    with pytest.raises(ValueError, match='invalid manifest'):
        apply_checkpoint_retention(
            tmp_path,
            RetentionConfiguration(
                recent_checkpoint_count=1,
                milestone_every_optimizer_steps=100,
                retain_replay_shards=True,
                retain_search_traces=False,
                retain_raw_evaluation_games=False,
            ),
        )

    assert tuple(sorted(path.name for path in tmp_path.glob('checkpoint-*') if path.is_dir())) == before


def test_retention_authenticates_current_pointer_before_deletion(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    for version in range(1, 3):
        repository.publish(
            _state(version),
            CheckpointPurpose.CREDIT_COMMIT,
            f'model-{version}'.encode(),
            f'optimizer-{version}'.encode(),
            f'random-{version}'.encode(),
        )
    pointer = CheckpointPointer.model_validate_json(repository.pointer_path.read_bytes())
    tampered = CheckpointPointer(
        schema_version=pointer.schema_version,
        run_id=pointer.run_id,
        model_version=pointer.model_version,
        checkpoint_directory=pointer.checkpoint_directory,
        manifest_sha256='f' * 64,
    )
    repository.pointer_path.write_text(tampered.model_dump_json(), encoding='utf-8')
    before = tuple(sorted(path.name for path in tmp_path.glob('checkpoint-*') if path.is_dir()))

    with pytest.raises(ValueError, match='authenticate'):
        apply_checkpoint_retention(
            tmp_path,
            RetentionConfiguration(
                recent_checkpoint_count=1,
                milestone_every_optimizer_steps=100,
                retain_replay_shards=True,
                retain_search_traces=False,
                retain_raw_evaluation_games=False,
            ),
        )

    assert tuple(sorted(path.name for path in tmp_path.glob('checkpoint-*') if path.is_dir())) == before
