from __future__ import annotations

import math
from pathlib import Path

from pydantic import model_validator
from src.search_stopping.configuration import SearchStoppingConfiguration
from src.util.frozen_model import FrozenModel


class SearchStopPolicy(FrozenModel):
    checkpoint_multiples: tuple[float, ...]
    thresholds: tuple[float, ...]
    movement_guard_epsilon: float
    cap_multiple: float
    predictor_path: Path | None
    predictor_sha256: str | None
    apply_learned: bool

    @model_validator(mode='after')
    def validate_policy(self) -> SearchStopPolicy:
        if len(self.thresholds) != len(self.checkpoint_multiples):
            raise ValueError('A stop policy requires one threshold per checkpoint multiple.')
        if any(not math.isfinite(multiple) or multiple <= 0.0 for multiple in self.checkpoint_multiples):
            raise ValueError('Stop-policy checkpoint multiples must be finite and positive.')
        for index in range(1, len(self.checkpoint_multiples)):
            if self.checkpoint_multiples[index] <= self.checkpoint_multiples[index - 1]:
                raise ValueError('Stop-policy checkpoint multiples must be strictly increasing.')
        if not math.isfinite(self.cap_multiple) or self.cap_multiple <= 1.0:
            raise ValueError('The stop-policy cap multiple must be finite and above one.')
        if self.checkpoint_multiples and self.checkpoint_multiples[-1] >= self.cap_multiple:
            raise ValueError('Stop-policy checkpoints must lie strictly below the cap multiple.')
        if any(not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0 for threshold in self.thresholds):
            raise ValueError('Stop thresholds must be probabilities in [0, 1].')
        if not math.isfinite(self.movement_guard_epsilon) or self.movement_guard_epsilon <= 0.0:
            raise ValueError('The movement guard epsilon must be finite and positive.')
        if (self.predictor_path is None) != (self.predictor_sha256 is None):
            raise ValueError('A stop-predictor reference requires both its path and its digest.')
        if self.predictor_sha256 is not None and not _is_sha256(self.predictor_sha256):
            raise ValueError('A stop-predictor digest must be 64 lowercase hex characters.')
        if self.apply_learned and self.predictor_path is None:
            raise ValueError('An applied stop policy requires a published predictor.')
        return self


def _is_sha256(digest: str) -> bool:
    return len(digest) == 64 and all(character in '0123456789abcdef' for character in digest)


def flat_stop_policy() -> SearchStopPolicy:
    """A closed policy with no checkpoint set: for evaluation and other always-flat searches."""
    return SearchStopPolicy(
        checkpoint_multiples=(),
        thresholds=(),
        movement_guard_epsilon=1e-3,
        cap_multiple=2.0,
        predictor_path=None,
        predictor_sha256=None,
        apply_learned=False,
    )


def closed_policy(configuration: SearchStoppingConfiguration) -> SearchStopPolicy:
    """The one fail-closed state: a flat search to the baseline — no checkpoints, no cap, no 2x burn."""
    return SearchStopPolicy(
        checkpoint_multiples=tuple(configuration.checkpoint_multiples),
        thresholds=(0.0,) * len(configuration.checkpoint_multiples),
        movement_guard_epsilon=configuration.movement_guard_epsilon,
        cap_multiple=configuration.cap_multiple,
        predictor_path=None,
        predictor_sha256=None,
        apply_learned=False,
    )


def checkpoint_visit_counts(checkpoint_multiples: tuple[float, ...], baseline_new_visits: int) -> tuple[int, ...]:
    if baseline_new_visits <= 0:
        raise ValueError('Baseline new visits must be positive.')
    visits = tuple(max(1, int(math.floor(multiple * baseline_new_visits + 0.5))) for multiple in checkpoint_multiples)
    if len(set(visits)) != len(visits):
        raise ValueError('Checkpoint multiples collapse to duplicate visit counts at this baseline.')
    return visits


def cap_visit_count(cap_multiple: float, baseline_new_visits: int) -> int:
    if baseline_new_visits <= 0:
        raise ValueError('Baseline new visits must be positive.')
    return int(math.floor(cap_multiple * baseline_new_visits + 0.5))
