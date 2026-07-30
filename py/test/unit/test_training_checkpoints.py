from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from src.az.training import checkpoints
from src.az.config.runtime import RetentionConfiguration
from src.az.experiment.artifact_retention import apply_checkpoint_retention
from src.az.replay.credits import EMPTY_CREDIT_PREFIX_SHA256, ReplayCreditState
from src.az.replay.sampling import ReplaySamplerState
from src.az.self_play.model_refresh import load_newer_model_checkpoint
from src.az.training.checkpoints import (
    CheckpointArtifact,
    CheckpointArtifactKind,
    CheckpointPointer,
    CheckpointPurpose,
    CheckpointRepository,
    DistributedCheckpointManifest,
    LoadedCheckpoint,
    TrainerCheckpointState,
)
from src.az.training.distributed import (
    DistributedBackend,
    ProcessGroupLifecycle,
    TrainingDeterminism,
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
        process_group=ProcessGroupLifecycle(
            backend=DistributedBackend.GLOO,
            rank=0,
            world_size=1,
            initialized=False,
        ),
        training_determinism=TrainingDeterminism.STRICT,
    )


def _repository(path: Path) -> CheckpointRepository:
    return CheckpointRepository(path.resolve(), UUID(int=1), CONFIGURATION_SHA256)


def _publish(
    repository: CheckpointRepository,
    state: TrainerCheckpointState,
    purpose: CheckpointPurpose,
    model: bytes,
    optimizer: bytes,
    random_state: bytes,
) -> LoadedCheckpoint:
    return repository.publish(
        state,
        purpose,
        model,
        optimizer,
        random_state,
        b'cuda-random',
        b'gradient-scaler',
    )


def _distributed_state(model_version: int, rank: int, world_size: int = 2) -> TrainerCheckpointState:
    common = _state(model_version)
    return TrainerCheckpointState(
        replay_credits=common.replay_credits,
        replay_sampler=common.replay_sampler,
        learning_rate=common.learning_rate,
        process_group=ProcessGroupLifecycle(
            backend=DistributedBackend.GLOO,
            rank=rank,
            world_size=world_size,
            initialized=True,
        ),
        training_determinism=TrainingDeterminism.SEEDED_CONCURRENT,
    )


def _stage_and_commit_distributed(
    repository: CheckpointRepository,
    model_version: int,
) -> None:
    for rank in range(2):
        repository.stage_distributed_rank(
            state=_distributed_state(model_version, rank),
            model_artifact=f'model-{model_version}'.encode(),
            optimizer_artifact=f'optimizer-{model_version}'.encode(),
            torch_random_state_artifact=f'random-{model_version}-{rank}'.encode(),
            cuda_random_stream_artifact=f'cuda-{model_version}-{rank}'.encode(),
            gradient_scaler_artifact=f'scaler-{model_version}'.encode(),
        )
    repository.commit_distributed_generation(
        state=_distributed_state(model_version, 0),
        purpose=CheckpointPurpose.SCHEDULED,
        model_artifact=f'model-{model_version}'.encode(),
        optimizer_artifact=f'optimizer-{model_version}'.encode(),
        gradient_scaler_artifact=f'scaler-{model_version}'.encode(),
    )


def test_checkpoint_publication_round_trips_opaque_artifacts(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    published = _publish(
        repository,
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
    original = _publish(
        repository,
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
    published = _publish(
        repository,
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


def test_authenticated_model_only_load_does_not_read_optimizer_state(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    published = _publish(
        repository,
        _state(1),
        CheckpointPurpose.CREDIT_COMMIT,
        b'model-1',
        b'optimizer-1',
        b'random-1',
    )
    checkpoint_directory = tmp_path / f'checkpoint-0000000001-{published.manifest.checkpoint_id.hex}'
    (checkpoint_directory / published.manifest.optimizer.filename).write_bytes(b'tampered optimizer')

    loaded = repository.load_current_model()

    assert loaded.model_artifact == b'model-1'
    with pytest.raises(ValueError, match='optimizer artifact checksum mismatch'):
        repository.load_current()


def test_model_refresh_accepts_newer_authenticated_pointer_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    _publish(
        repository,
        _state(1),
        CheckpointPurpose.CREDIT_COMMIT,
        b'model-1',
        b'optimizer-1',
        b'random-1',
    )

    def advance_after_probe() -> int:
        _publish(
            repository,
            _state(2),
            CheckpointPurpose.SCHEDULED,
            b'model-2',
            b'optimizer-2',
            b'random-2',
        )
        return 1

    monkeypatch.setattr(repository, 'current_model_version', advance_after_probe)
    loaded = load_newer_model_checkpoint(repository, current_model_version=0)

    assert loaded is not None
    assert loaded.manifest.model_version == 2
    assert loaded.model_artifact == b'model-2'


def test_missing_distributed_rank_does_not_advance_global_pointer(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    original = _publish(
        repository,
        _state(1),
        CheckpointPurpose.CREDIT_COMMIT,
        b'model-1',
        b'optimizer-1',
        b'random-1',
    )
    distributed_state = _state(2).model_copy(
        update={
            'process_group': ProcessGroupLifecycle(
                backend=DistributedBackend.GLOO,
                rank=0,
                world_size=2,
                initialized=True,
            )
        }
    )
    repository.stage_distributed_rank(
        state=distributed_state,
        model_artifact=b'model-2',
        optimizer_artifact=b'optimizer-2',
        torch_random_state_artifact=b'random-rank-0',
        cuda_random_stream_artifact=b'cuda-rank-0',
        gradient_scaler_artifact=b'scaler-2',
    )

    with pytest.raises(ValueError, match='rank 1 has not staged'):
        repository.commit_distributed_generation(
            state=distributed_state,
            purpose=CheckpointPurpose.SCHEDULED,
            model_artifact=b'model-2',
            optimizer_artifact=b'optimizer-2',
            gradient_scaler_artifact=b'scaler-2',
        )

    assert repository.load_current() == original
    assert repository.current_model_version() == 1


def test_distributed_commit_recovers_rename_before_pointer_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    for rank in range(2):
        repository.stage_distributed_rank(
            state=_distributed_state(1, rank),
            model_artifact=b'model-1',
            optimizer_artifact=b'optimizer-1',
            torch_random_state_artifact=f'random-{rank}'.encode(),
            cuda_random_stream_artifact=f'cuda-{rank}'.encode(),
            gradient_scaler_artifact=b'scaler-1',
        )
    atomic_replace = checkpoints._atomic_replace

    def fail_pointer_replace(path: Path, contents: bytes) -> None:
        if path == repository.pointer_path:
            raise OSError('injected pointer crash')
        atomic_replace(path, contents)

    monkeypatch.setattr(checkpoints, '_atomic_replace', fail_pointer_replace)
    with pytest.raises(OSError, match='injected pointer crash'):
        repository.commit_distributed_generation(
            state=_distributed_state(1, 0),
            purpose=CheckpointPurpose.SCHEDULED,
            model_artifact=b'model-1',
            optimizer_artifact=b'optimizer-1',
            gradient_scaler_artifact=b'scaler-1',
        )
    assert (tmp_path / 'distributed-0000000001').is_dir()
    assert not repository.pointer_path.exists()

    monkeypatch.setattr(checkpoints, '_atomic_replace', atomic_replace)
    recovered = repository.commit_distributed_generation(
        state=_distributed_state(1, 0),
        purpose=CheckpointPurpose.SCHEDULED,
        model_artifact=b'model-1',
        optimizer_artifact=b'optimizer-1',
        gradient_scaler_artifact=b'scaler-1',
    )

    assert recovered.manifest.model_version == 1
    assert repository.current_model_version() == 1


def test_distributed_model_only_load_opens_only_model_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    _stage_and_commit_distributed(repository, 1)
    opened_kinds: list[CheckpointArtifactKind] = []
    read_artifact = CheckpointRepository._read_artifact

    def instrumented_read(
        self: CheckpointRepository,
        checkpoint_directory: Path,
        artifact: CheckpointArtifact,
    ) -> bytes:
        opened_kinds.append(artifact.kind)
        return read_artifact(self, checkpoint_directory, artifact)

    monkeypatch.setattr(CheckpointRepository, '_read_artifact', instrumented_read)
    loaded = repository.load_current_model()

    assert loaded.manifest.model_version == 1
    assert loaded.model_artifact == b'model-1'
    assert opened_kinds == [CheckpointArtifactKind.MODEL]


@pytest.mark.parametrize(
    ('field', 'message'),
    (
        ('rank_identity', 'rank identity'),
        ('artifact_role', 'common artifacts'),
        ('created_at', 'timezone-aware UTC'),
    ),
)
def test_distributed_manifest_rejects_inconsistent_identity_and_roles(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    repository = _repository(tmp_path)
    _stage_and_commit_distributed(repository, 1)
    candidate = repository.load_distributed(0).manifest.model_dump()
    if field == 'rank_identity':
        candidate['ranks'][1]['state']['process_group']['rank'] = 0
    elif field == 'artifact_role':
        candidate['model']['kind'] = CheckpointArtifactKind.OPTIMIZER
    else:
        candidate['created_at'] = datetime(2026, 1, 1)

    with pytest.raises(ValueError, match=message):
        DistributedCheckpointManifest.model_validate(candidate)


def test_failed_pointer_commit_leaves_previous_credit_state_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    original = _publish(
        repository,
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
        _publish(
            repository,
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
        loaded = _publish(
            repository,
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


def test_retention_prunes_authenticated_distributed_generations(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _stage_and_commit_distributed(repository, 1)
    _stage_and_commit_distributed(repository, 2)

    result = apply_checkpoint_retention(
        tmp_path,
        RetentionConfiguration(
            recent_checkpoint_count=1,
            milestone_every_optimizer_steps=1_000,
            retain_replay_shards=True,
            retain_search_traces=False,
            retain_raw_evaluation_games=False,
        ),
    )

    assert result.retained_checkpoint_directories == ('distributed-0000000002',)
    assert result.deleted_checkpoint_directories == ('distributed-0000000001',)
    assert repository.load_distributed(0).manifest.model_version == 2


def test_distributed_retention_preflight_prevents_partial_deletion(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _stage_and_commit_distributed(repository, 1)
    _stage_and_commit_distributed(repository, 2)
    (tmp_path / 'distributed-0000000001' / 'optimizer.pt').write_bytes(b'corrupt')

    with pytest.raises(ValueError, match='artifact is corrupt'):
        apply_checkpoint_retention(
            tmp_path,
            RetentionConfiguration(
                recent_checkpoint_count=1,
                milestone_every_optimizer_steps=1_000,
                retain_replay_shards=True,
                retain_search_traces=False,
                retain_raw_evaluation_games=False,
            ),
        )

    assert (tmp_path / 'distributed-0000000001').is_dir()
    assert (tmp_path / 'distributed-0000000002').is_dir()


def test_retention_preflight_prevents_partial_deletion_on_corrupt_old_manifest(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    for version in range(1, 4):
        _publish(
            repository,
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
        _publish(
            repository,
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
