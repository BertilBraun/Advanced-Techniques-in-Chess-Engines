from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from src.search_stopping.features import (
    STOP_PREDICTOR_FEATURE_COUNT,
    CheckpointFeatureContext,
    checkpoint_feature_vector,
)
from src.search_stopping.labels import CheckpointObservation
from src.search_stopping.records import append_audit_records, audit_record_dtype, read_audit_records
from src.search_stopping.sampling import AuditPositionIdentity, is_audit_position
from src.search_stopping.targets import PolicyDistribution


def _policy(*probabilities: float) -> PolicyDistribution:
    return PolicyDistribution(probabilities=probabilities)


def _context(starting_visits: int = 0) -> CheckpointFeatureContext:
    return CheckpointFeatureContext(
        prior=_policy(0.5, 0.3, 0.2),
        network_root_value=0.1,
        ply=40,
        baseline_visits=800,
        model_generation=365,
        starting_visits=starting_visits,
    )


def test_feature_vector_matches_golden_values() -> None:
    current = CheckpointObservation(visits=400, root_value=0.25, policy=_policy(0.6, 0.3, 0.1))
    previous = CheckpointObservation(visits=200, root_value=0.15, policy=_policy(0.5, 0.4, 0.1))
    vector = checkpoint_feature_vector(current, previous, _context(starting_visits=240), 0.5)
    assert len(vector) == STOP_PREDICTOR_FEATURE_COUNT
    assert vector[0] == pytest.approx(0.6)  # top share
    assert vector[1] == pytest.approx(-(0.6 * math.log(0.6) + 0.3 * math.log(0.3) + 0.1 * math.log(0.1)))
    assert vector[2] == pytest.approx(0.3)  # top-two gap
    assert vector[4] == pytest.approx(
        0.6 * math.log(0.6 / 0.5) + 0.3 * math.log(0.3 / 0.4)  # movement KL, third term is zero
    )
    # segment over raw counts: (400*pi_now - 200*pi_prev) / 200 -> (0.7, 0.2, 0.1)
    assert vector[5] == pytest.approx(0.7)
    assert vector[7] == pytest.approx(0.1)  # value trend
    assert vector[8] == pytest.approx(0.15)  # value minus network
    assert vector[15] == pytest.approx(0.5)  # checkpoint multiple
    assert vector[16] == pytest.approx(0.3)  # root warmth 240/800
    assert vector[17] == pytest.approx(3.0)  # support count
    assert vector[18] == pytest.approx(0.1)  # top-3 share


def test_fresh_root_uses_the_prior_as_previous_distribution() -> None:
    current = CheckpointObservation(visits=400, root_value=0.25, policy=_policy(0.5, 0.3, 0.2))
    vector = checkpoint_feature_vector(current, None, _context(), 0.5)
    assert vector[4] == pytest.approx(0.0)  # no movement relative to the prior
    assert vector[7] == pytest.approx(0.25 - 0.1)  # value trend against the network value


def test_audit_records_round_trip(tmp_path: Path) -> None:
    dtype = audit_record_dtype(5)
    records = np.zeros(3, dtype=dtype)
    records['ply'] = (10, 20, 30)
    records['kl_to_final'][:, 2] = 0.5
    path = tmp_path / 'audit-generation-00000001.np'
    append_audit_records(path, records, checkpoint_count=5)
    append_audit_records(path, records, checkpoint_count=5)
    loaded = read_audit_records(path, checkpoint_count=5)
    assert loaded.shape == (6,)
    assert loaded['kl_to_final'][0, 2] == pytest.approx(0.5)


def test_audit_records_reject_a_mismatched_dtype(tmp_path: Path) -> None:
    records = np.zeros(1, dtype=audit_record_dtype(4))
    with pytest.raises(ValueError, match='dtype'):
        append_audit_records(tmp_path / 'audit.np', records, checkpoint_count=5)


def test_audit_sampling_is_deterministic_and_near_the_requested_fraction() -> None:
    fraction = Fraction(1, 50)
    identities = tuple(
        AuditPositionIdentity(source_generation=7, game_identity=f'game-{game}', ply=ply)
        for game in range(200)
        for ply in range(25)
    )
    selected = tuple(identity for identity in identities if is_audit_position(identity, 99, fraction))
    reselected = tuple(identity for identity in identities if is_audit_position(identity, 99, fraction))
    assert selected == reselected
    assert 0.5 * float(fraction) < len(selected) / len(identities) < 2.0 * float(fraction)


def test_audit_sampling_changes_with_the_run_seed() -> None:
    identity = AuditPositionIdentity(source_generation=7, game_identity='game-1', ply=3)
    verdicts = {is_audit_position(identity, seed, Fraction(1, 2)) for seed in range(32)}
    assert verdicts == {True, False}
