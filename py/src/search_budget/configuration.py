from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator
from src.search_budget.curve import BLEND_CANDIDATES
from src.util.frozen_model import FrozenModel


class DeepLabelingConfiguration(FrozenModel):
    sample_fraction: Decimal = Field(default=Decimal('0.02'), gt=Decimal(0), le=Decimal(1))
    deep_search_multiple: int = Field(default=8, gt=1)
    maximum_unstarted_generation_lag: int = Field(default=2, ge=0)
    persisted_shard_size: int = Field(default=512, gt=0, le=512)
    inference_workers_per_process: int = Field(default=1, gt=0)
    inference_batch_size: int = Field(default=512, gt=0, le=512)
    outstanding_inference_batches: int = Field(default=2, ge=1, le=2)
    parallel_searches: int = Field(default=2, gt=0)

    @model_validator(mode='after')
    def validate_first_run_defaults(self) -> DeepLabelingConfiguration:
        if self.deep_search_multiple != 8:
            raise ValueError('Deep search must use exactly eight times the source-generation baseline.')
        if self.persisted_shard_size != 512 or self.inference_batch_size != 512:
            raise ValueError('Deep-label shards and inference batches must use the settled size of 512.')
        if self.inference_workers_per_process != 1 or self.outstanding_inference_batches != 2:
            raise ValueError('Each label process must own one inference worker with two outstanding batches.')
        if self.parallel_searches != 2:
            raise ValueError('Deep-label search parallelism must remain two.')
        return self


class BlendCalibrationConfiguration(FrozenModel):
    candidate_blends: tuple[Decimal, ...] = BLEND_CANDIDATES
    warmup_completed_source_generations: int = Field(default=30, gt=0)
    ema_decay: Decimal = Field(default=Decimal('0.2'), gt=Decimal(0), le=Decimal(1))
    maximum_upward_step: Decimal = Field(default=Decimal('0.1'), gt=Decimal(0), le=Decimal(1))

    @model_validator(mode='after')
    def validate_first_run_defaults(self) -> BlendCalibrationConfiguration:
        if self.candidate_blends != BLEND_CANDIDATES:
            raise ValueError('Blend candidates must be exactly [0.0, 0.1, ..., 1.0].')
        if self.warmup_completed_source_generations != 30:
            raise ValueError('Allocator warm-up must contain exactly 30 completed source generations.')
        if self.ema_decay != Decimal('0.2') or self.maximum_upward_step != Decimal('0.1'):
            raise ValueError('Blend calibration must use EMA decay 0.2 and maximum upward step 0.1.')
        return self


class ProductionAllocationConfiguration(FrozenModel):
    sequential_round_target: int = Field(default=200, gt=0)
    maximum_parallel_searches: int = Field(default=16, gt=0)

    @model_validator(mode='after')
    def validate_first_run_defaults(self) -> ProductionAllocationConfiguration:
        if self.sequential_round_target != 200 or self.maximum_parallel_searches != 16:
            raise ValueError('Production allocation must target 200 rounds and cap parallel searches at 16.')
        return self


class SearchBudgetConfiguration(FrozenModel):
    curve_version: Literal['measured_oracle_600_v1'] = 'measured_oracle_600_v1'
    labeling: DeepLabelingConfiguration = DeepLabelingConfiguration()
    calibration: BlendCalibrationConfiguration = BlendCalibrationConfiguration()
    production: ProductionAllocationConfiguration = ProductionAllocationConfiguration()
