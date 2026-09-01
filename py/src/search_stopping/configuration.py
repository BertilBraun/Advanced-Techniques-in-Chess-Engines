from __future__ import annotations

import math
from decimal import Decimal

from pydantic import Field, model_validator
from src.util.frozen_model import FrozenModel


class SearchStoppingConfiguration(FrozenModel):
    audit_sample_fraction: Decimal
    anchor_fraction: Decimal
    anchor_visit_multiple: float = Field(gt=1.0)
    checkpoint_multiples: tuple[float, ...] = Field(min_length=1)
    cap_multiple: float = Field(gt=1.0)
    eps_pi_minimum: float = Field(gt=0.0)
    eps_pi_maximum: float = Field(gt=0.0)
    eps_v: float = Field(gt=0.0)
    movement_guard_epsilon: float = Field(gt=0.0)
    false_stop_rate_ceiling: float = Field(gt=0.0, lt=1.0)
    minimum_evidence_trigger_count: int = Field(gt=0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    first_production_generation: int = Field(ge=0)
    maximum_realized_mean_spend: float = Field(gt=1.0)
    window_generations: int = Field(gt=0)
    maximum_unstarted_generation_lag: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_stopping(self) -> SearchStoppingConfiguration:
        if not Decimal(0) < self.audit_sample_fraction <= Decimal(1):
            raise ValueError('The audit sample fraction must lie in (0, 1].')
        if not Decimal(0) <= self.anchor_fraction <= Decimal(1):
            raise ValueError('The anchor fraction must lie in [0, 1].')
        if any(not math.isfinite(multiple) or multiple <= 0.0 for multiple in self.checkpoint_multiples):
            raise ValueError('Checkpoint multiples must be finite and positive.')
        for index in range(1, len(self.checkpoint_multiples)):
            if self.checkpoint_multiples[index] <= self.checkpoint_multiples[index - 1]:
                raise ValueError('Checkpoint multiples must be strictly increasing.')
        if not math.isfinite(self.cap_multiple):
            raise ValueError('The cap multiple must be finite.')
        if self.checkpoint_multiples[-1] >= self.cap_multiple:
            raise ValueError('Every checkpoint multiple must lie strictly below the cap multiple.')
        if not math.isfinite(self.anchor_visit_multiple) or self.anchor_visit_multiple <= self.cap_multiple:
            raise ValueError('The anchor visit multiple must lie above the cap multiple.')
        if not math.isfinite(self.eps_pi_minimum) or not math.isfinite(self.eps_pi_maximum):
            raise ValueError('The eps clamp must be finite.')
        if self.eps_pi_maximum <= self.eps_pi_minimum:
            raise ValueError('The eps clamp maximum must lie above its minimum.')
        if not math.isfinite(self.eps_v):
            raise ValueError('The value epsilon must be finite.')
        if not math.isfinite(self.movement_guard_epsilon):
            raise ValueError('The movement guard epsilon must be finite.')
        if not math.isfinite(self.maximum_realized_mean_spend) or self.maximum_realized_mean_spend > self.cap_multiple:
            raise ValueError('The spend circuit-breaker limit must be finite and at most the cap multiple.')
        return self
