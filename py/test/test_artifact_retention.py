import hashlib
from pathlib import Path

from src.experiment.artifact_retention import apply_artifact_retention
from src.train.TrainingArgs import ArtifactRetention
from src.util.save_paths import CheckpointManifest, ReplayFileReference, load_checkpoint_manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_checkpoint(root: Path, iteration: int) -> None:
    model_path = root / f'model_{iteration}.pt'
    optimizer_path = root / f'optimizer_{iteration}.pt'
    jit_model_path = root / f'model_{iteration}.jit.pt'
    model_path.write_bytes(f'model-{iteration}'.encode())
    optimizer_path.write_bytes(f'optimizer-{iteration}'.encode())
    jit_model_path.write_bytes(f'jit-{iteration}'.encode())

    replay_files = tuple(
        ReplayFileReference(
            path=f'memory_{replay_iteration}/sample.hdf5',
            size_bytes=(root / f'memory_{replay_iteration}' / 'sample.hdf5').stat().st_size,
        )
        for replay_iteration in range(iteration + 1)
    )
    manifest = CheckpointManifest(
        iteration=iteration,
        model_path=model_path.name,
        model_sha256=sha256(model_path),
        optimizer_path=optimizer_path.name,
        optimizer_sha256=sha256(optimizer_path),
        jit_model_path=jit_model_path.name,
        jit_model_sha256=sha256(jit_model_path),
        replay_files=replay_files,
    )
    (root / f'checkpoint_{iteration}.json').write_text(
        manifest.model_dump_json(indent=2),
        encoding='utf-8',
    )


def test_retention_keeps_five_resumable_checkpoints_and_bounded_replay(tmp_path: Path) -> None:
    for iteration in range(11):
        replay_path = tmp_path / f'memory_{iteration}' / 'sample.hdf5'
        replay_path.parent.mkdir()
        replay_path.write_bytes(f'replay-{iteration}'.encode())
        write_checkpoint(tmp_path, iteration)

    result = apply_artifact_retention(
        tmp_path,
        latest_checkpoint_iteration=10,
        retention=ArtifactRetention(
            checkpoint_count=5,
            replay_window_iterations=3,
        ),
    )

    assert result.earliest_checkpoint_iteration == 6
    assert result.earliest_replay_iteration == 3
    assert result.deleted_checkpoint_files == 24
    assert result.deleted_replay_directories == 3
    assert {path.name for path in tmp_path.glob('checkpoint_*.json')} == {
        f'checkpoint_{iteration}.json' for iteration in range(6, 11)
    }
    assert {path.name for path in tmp_path.glob('memory_*')} == {f'memory_{iteration}' for iteration in range(3, 11)}
    for iteration in range(6, 11):
        manifest = load_checkpoint_manifest(iteration, tmp_path)
        assert len(manifest.replay_files) == 4
