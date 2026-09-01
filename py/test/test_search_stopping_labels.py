from __future__ import annotations

import pytest
from src.search_stopping.labels import (
    CappedSearchRecord,
    CheckpointObservation,
    checkpoint_stop_labels,
)
from src.search_stopping.targets import PolicyDistribution, policy_kl


def _policy(*probabilities: float) -> PolicyDistribution:
    return PolicyDistribution(probabilities=probabilities)


def _record(checkpoint_policies: tuple[PolicyDistribution, ...], final: PolicyDistribution) -> CappedSearchRecord:
    return CappedSearchRecord(
        checkpoints=tuple(
            CheckpointObservation(visits=100 * (index + 1), root_value=0.0, policy=policy)
            for index, policy in enumerate(checkpoint_policies)
        ),
        final_visits=100 * (len(checkpoint_policies) + 1),
        final_root_value=0.0,
        final_policy=final,
    )


FINAL = _policy(0.7, 0.2, 0.1)
NEAR_FINAL = _policy(0.69, 0.21, 0.1)
FAR_FROM_FINAL = _policy(0.1, 0.2, 0.7)


def test_converged_checkpoints_are_certain() -> None:
    labels = checkpoint_stop_labels(_record((NEAR_FINAL, NEAR_FINAL), FINAL), eps_pi=0.05, eps_v=0.5)
    assert [label.uncertain for label in labels] == [False, False]


def test_future_max_clause_marks_early_checkpoint_uncertain_when_a_later_one_diverges() -> None:
    labels = checkpoint_stop_labels(_record((NEAR_FINAL, FAR_FROM_FINAL), FINAL), eps_pi=0.05, eps_v=0.5)
    assert [label.uncertain for label in labels] == [True, True]


def test_divergence_before_a_checkpoint_does_not_affect_it() -> None:
    labels = checkpoint_stop_labels(_record((FAR_FROM_FINAL, NEAR_FINAL), FINAL), eps_pi=0.05, eps_v=0.5)
    assert [label.uncertain for label in labels] == [True, False]


def test_divergence_exactly_at_eps_is_uncertain() -> None:
    divergence = policy_kl(FINAL, NEAR_FINAL)
    labels = checkpoint_stop_labels(_record((NEAR_FINAL,), FINAL), eps_pi=divergence, eps_v=0.5)
    assert labels[0].uncertain


def test_value_drift_alone_marks_uncertain() -> None:
    record = CappedSearchRecord(
        checkpoints=(CheckpointObservation(visits=100, root_value=0.4, policy=NEAR_FINAL),),
        final_visits=200,
        final_root_value=0.0,
        final_policy=FINAL,
    )
    labels = checkpoint_stop_labels(record, eps_pi=0.05, eps_v=0.3)
    assert labels[0].uncertain and labels[0].value_gap == pytest.approx(0.4)


def test_argmax_swap_is_recorded_without_forcing_uncertainty() -> None:
    swapped = _policy(0.45, 0.45001, 0.09999)
    near_swapped = _policy(0.45001, 0.45, 0.09999)
    labels = checkpoint_stop_labels(_record((swapped,), near_swapped), eps_pi=1.0, eps_v=1.0)
    assert labels[0].argmax_swap and not labels[0].uncertain


def test_checkpoints_must_precede_the_final_visit_count() -> None:
    with pytest.raises(ValueError, match='strictly below'):
        CappedSearchRecord(
            checkpoints=(CheckpointObservation(visits=200, root_value=0.0, policy=NEAR_FINAL),),
            final_visits=200,
            final_root_value=0.0,
            final_policy=FINAL,
        )


@pytest.mark.parametrize('eps_pi,eps_v', [(0.0, 0.5), (-1.0, 0.5), (0.05, 0.0), (float('nan'), 0.5)])
def test_labels_reject_non_positive_epsilons(eps_pi: float, eps_v: float) -> None:
    with pytest.raises(ValueError):
        checkpoint_stop_labels(_record((NEAR_FINAL,), FINAL), eps_pi=eps_pi, eps_v=eps_v)
