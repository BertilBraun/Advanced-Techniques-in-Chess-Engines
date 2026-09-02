from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
from src.search_stopping.configuration import SearchStoppingConfiguration
from src.search_stopping.manager import SearchStoppingManager
from src.search_stopping.records import (
    PAIRED_FLOOR_RECORD_DTYPE,
    append_records,
    audit_log_path,
    audit_record_dtype,
    paired_floor_log_path,
)

CONFIGURATION_SHA = 'a' * 64
CHECKPOINT_COUNT = 5


def _configuration() -> SearchStoppingConfiguration:
    return SearchStoppingConfiguration(
        audit_sample_fraction=Decimal('0.01'),
        paired_audit_fraction=Decimal('0.1'),
        noise_floor_multiple=1.0,
        anchor_fraction=Decimal('0.05'),
        anchor_visit_multiple=4.0,
        checkpoint_multiples=(1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0, 1.5),
        cap_multiple=2.0,
        eps_pi_minimum=0.02,
        eps_pi_maximum=0.3,
        eps_v=0.3,
        movement_guard_epsilon=0.05,
        excess_cost_ceiling=0.25,
        catastrophic_excess_multiple=5.0,
        catastrophic_stop_ceiling=0.01,
        minimum_evidence_trigger_count=100,
        confidence_level=0.95,
        first_production_generation=10,
        maximum_realized_mean_spend=1.3,
        window_generations=10,
        maximum_unstarted_generation_lag=2,
    )


def _manager(run_path: Path, worker_count: int = 4) -> SearchStoppingManager:
    return SearchStoppingManager(
        run_path=run_path,
        configuration=_configuration(),
        configuration_sha256=CONFIGURATION_SHA,
        first_unstarted_production_generation=0,
        worker_count=worker_count,
    )


def _write_audit(run_path: Path, generation: int, worker_id: int, baseline_visits: int, plies: tuple[int, ...]) -> None:
    dtype = audit_record_dtype(CHECKPOINT_COUNT)
    records = np.zeros(len(plies), dtype=dtype)
    records['source_generation'] = generation
    records['ply'] = plies
    records['baseline_visits'] = baseline_visits
    records['game_key'] = worker_id + 1
    append_records(audit_log_path(run_path / 'search-stopping', generation, worker_id), records, dtype)


def _write_floor(run_path: Path, generation: int, worker_id: int, kl_symmetric: float) -> None:
    record = np.zeros(1, dtype=PAIRED_FLOOR_RECORD_DTYPE)
    record['source_generation'] = generation
    record['kl_symmetric'] = kl_symmetric
    append_records(
        paired_floor_log_path(run_path / 'search-stopping', generation, worker_id), record, PAIRED_FLOOR_RECORD_DTYPE
    )


def test_cached_window_matches_a_fresh_read(tmp_path: Path) -> None:
    for generation in range(3):
        for worker_id in range(4):
            _write_audit(tmp_path, generation, worker_id, 400, (generation, worker_id))
            _write_floor(tmp_path, generation, worker_id, 0.01 * (worker_id + 1))
    cached = _manager(tmp_path)
    cached._load_audit_window([0, 1])
    cached._load_paired_floors([0, 1])
    fresh = _manager(tmp_path)

    window = [0, 1, 2]
    assert np.array_equal(cached._load_audit_window(window), fresh._load_audit_window(window))
    assert np.array_equal(cached._load_paired_floors(window), fresh._load_paired_floors(window))


def test_window_still_breaks_at_a_baseline_visits_step(tmp_path: Path) -> None:
    _write_audit(tmp_path, 0, 0, 300, (1, 2))
    _write_audit(tmp_path, 1, 0, 400, (3,))
    _write_audit(tmp_path, 2, 0, 400, (4,))
    manager = _manager(tmp_path)

    records = manager._load_audit_window([0, 1, 2])

    assert records is not None
    assert sorted(records['ply'].tolist()) == [3, 4]


def test_window_ignores_a_missing_generation(tmp_path: Path) -> None:
    _write_audit(tmp_path, 0, 0, 400, (1,))
    _write_audit(tmp_path, 2, 0, 400, (2,))
    manager = _manager(tmp_path)

    records = manager._load_audit_window([0, 1, 2])

    assert records is not None
    assert sorted(records['ply'].tolist()) == [1, 2]


def test_glob_fallback_finds_files_beyond_the_worker_count(tmp_path: Path) -> None:
    _write_audit(tmp_path, 0, 7, 400, (9,))
    _write_floor(tmp_path, 0, 7, 0.02)
    manager = _manager(tmp_path, worker_count=4)

    records = manager._load_audit_window([0])
    floors = manager._load_paired_floors([0])

    assert records is not None and records['ply'].tolist() == [9]
    assert floors.tolist() == [np.float32(0.02)]


def test_cache_evicts_generations_outside_the_window(tmp_path: Path) -> None:
    for generation in range(4):
        _write_audit(tmp_path, generation, 0, 400, (generation,))
    manager = _manager(tmp_path)
    manager._load_audit_window([0, 1, 2])
    manager._load_audit_window([2, 3])

    assert sorted(manager._audit_cache) == [2, 3]
