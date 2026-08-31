from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator
from src.util.frozen_model import FrozenModel
from src.util.generation_schedule import ConstantSchedule, DecimalGenerationSchedule, defined_schedule_values


class LabelArtifactRetention(str, Enum):
    RETAIN_ALL = 'retain_all'
    REMOVE_BULKY_AFTER_FINALIZATION = 'remove_bulky_after_finalization'


class DeepLabelingConfiguration(FrozenModel):
    sample_fraction: DecimalGenerationSchedule = ConstantSchedule[Decimal](value=Decimal('0.02'))
    deep_search_multiple: int = Field(default=8, gt=1)
    maximum_unstarted_generation_lag: int = Field(default=2, ge=0)
    persisted_shard_size: int = Field(default=512, gt=0, le=512)
    inference_workers_per_process: int = Field(default=1, gt=0)
    inference_batch_size: int = Field(default=512, gt=0, le=512)
    outstanding_inference_batches: int = Field(default=2, ge=1, le=2)
    parallel_searches: int = Field(default=2, gt=0)
    artifact_retention: LabelArtifactRetention = LabelArtifactRetention.RETAIN_ALL

    @model_validator(mode='after')
    def validate_sample_fractions(self) -> DeepLabelingConfiguration:
        for value in defined_schedule_values(self.sample_fraction):
            if not Decimal(0) < value <= Decimal(1):
                raise ValueError('Every label sample fraction must lie in (0, 1].')
        return self


class BudgetCurveCorrectorConfiguration(FrozenModel):
    enabled: bool = True
    window_generations: int = Field(default=10, gt=0)


class BudgetPolicyCalibrationConfiguration(FrozenModel):
    warmup_completed_source_generations: int = Field(default=30, gt=0)
    sigma_ema_decay: Decimal = Field(default=Decimal('0.1'), gt=Decimal(0), le=Decimal(1))
    validation_gain_ema_decay: Decimal = Field(default=Decimal('0.2'), gt=Decimal(0), le=Decimal(1))
    lambda_trust_ratio: Decimal = Field(default=Decimal('2.0'), gt=Decimal(1))
    lambda_reseed_ratio: Decimal = Field(default=Decimal('100.0'), gt=Decimal(1))
    corrector: BudgetCurveCorrectorConfiguration = BudgetCurveCorrectorConfiguration()


class ProductionAllocationConfiguration(FrozenModel):
    sequential_round_target: int = Field(default=200, gt=0)
    minimum_parallel_searches: int = Field(default=2, gt=0)
    maximum_parallel_searches: int = Field(default=16, gt=0)


class SearchBudgetHeadTrainingConfiguration(FrozenModel):
    # Labels cover well under one percent of replay, so mixing them into ordinary batches leaves ~10 labelled rows
    # in 2048 and the head's telemetry measures sampling noise instead of learning.
    dedicated_batches: bool = True
    interval_optimizer_steps: int = Field(default=50, gt=0)


class SearchBudgetConfiguration(FrozenModel):
    curve_version: Literal['predicted_kl_curve_v1'] = 'predicted_kl_curve_v1'
    labeling: DeepLabelingConfiguration = DeepLabelingConfiguration()
    calibration: BudgetPolicyCalibrationConfiguration = BudgetPolicyCalibrationConfiguration()
    production: ProductionAllocationConfiguration = ProductionAllocationConfiguration()
    head_training: SearchBudgetHeadTrainingConfiguration = SearchBudgetHeadTrainingConfiguration()
