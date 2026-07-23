from pathlib import Path

from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.self_play.value_target import TerminationReason
from tools.production_ddp_fixture import PRODUCTION_GLOBAL_BATCH_SIZE, write_replay_fixture


def test_production_smoke_writes_schema_v2_replay(tmp_path: Path) -> None:
    replay_path = tmp_path / 'memory_0' / 'smoke.hdf5'

    write_replay_fixture(replay_path, sample_count=9, seed=17)
    replay = SelfPlayDataset.load_strict(replay_path)

    assert len(replay) == 9
    assert all(target.outcome_target_eligible for target in replay.value_targets)
    assert {target.termination_reason for target in replay.value_targets} == {TerminationReason.NATURAL}


def test_production_smoke_uses_1024_global_batch() -> None:
    assert PRODUCTION_GLOBAL_BATCH_SIZE == 1024
