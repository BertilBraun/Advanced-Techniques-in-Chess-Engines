from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator
from src.util.frozen_model import FrozenModel


class LabelArtifactRetention(str, Enum):
    RETAIN_ALL = 'retain_all'
    REMOVE_BULKY_AFTER_FINALIZATION = 'remove_bulky_after_finalization'


class DeepLabelingConfiguration(FrozenModel):
    sample_fraction: Decimal = Field(default=Decimal('0.02'), gt=Decimal(0), le=Decimal(1))
    deep_search_multiple: int = Field(default=8, gt=1)
    maximum_unstarted_generation_lag: int = Field(default=2, ge=0)
    persisted_shard_size: int = Field(default=512, gt=0, le=512)
    inference_workers_per_process: int = Field(default=1, gt=0)
    inference_batch_size: int = Field(default=512, gt=0, le=512)
    outstanding_inference_batches: int = Field(default=2, ge=1, le=2)
    parallel_searches: int = Field(default=2, gt=0)
    artifact_retention: LabelArtifactRetention = LabelArtifactRetention.RETAIN_ALL

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


class CurveCalibrationConfiguration(FrozenModel):
    bucket_count: int = Field(default=10, ge=2)
    initializer_version: Literal['analytic_q5_v1'] = 'analytic_q5_v1'
    warmup_completed_source_generations: int = Field(default=30, gt=0)
    bucket_utility_ema_decay: Decimal = Field(default=Decimal('0.2'), gt=Decimal(0), le=Decimal(1))
    validation_gain_ema_decay: Decimal = Field(default=Decimal('0.2'), gt=Decimal(0), le=Decimal(1))
    probe_ratio: Decimal = Field(default=Decimal('1.1'), gt=Decimal(1))
    maximum_step_ratio: Decimal = Field(default=Decimal('1.1'), gt=Decimal(1))

    @model_validator(mode='after')
    def validate_first_run_defaults(self) -> CurveCalibrationConfiguration:
        if self.bucket_count != 10 or self.initializer_version != 'analytic_q5_v1':
            raise ValueError('Live curve calibration must use ten buckets and the analytic q5 initializer.')
        if self.warmup_completed_source_generations != 30:
            raise ValueError('Allocator warm-up must contain exactly 30 completed source generations.')
        if self.bucket_utility_ema_decay != Decimal('0.2') or self.validation_gain_ema_decay != Decimal('0.2'):
            raise ValueError('Curve calibration must use EMA decay 0.2 for utility and validation gain.')
        if self.probe_ratio != Decimal('1.1') or self.maximum_step_ratio != Decimal('1.1'):
            raise ValueError('Curve calibration must use a 1.1 probe and maximum multiplicative step ratio.')
        return self


class ProductionAllocationConfiguration(FrozenModel):
    sequential_round_target: int = Field(default=200, gt=0)
    minimum_parallel_searches: int = Field(default=2, gt=0)
    maximum_parallel_searches: int = Field(default=16, gt=0)

    @model_validator(mode='after')
    def validate_first_run_defaults(self) -> ProductionAllocationConfiguration:
        if (
            self.sequential_round_target != 200
            or self.minimum_parallel_searches != 2
            or self.maximum_parallel_searches != 16
        ):
            raise ValueError('Production allocation must target 200 rounds and use parallel-search bounds 2..16.')
        return self


class SearchBudgetHeadTrainingConfiguration(FrozenModel):
    # Labels cover well under one percent of replay, so mixing them into ordinary batches leaves ~10 labelled rows
    # in 2048 and the head's telemetry measures sampling noise instead of learning.
    dedicated_batches: bool = True
    batch_size: int = Field(default=2000, gt=0)
    interval_optimizer_steps: int = Field(default=50, gt=0)
    minimum_labelled_rows: int = Field(default=256, gt=0)

    @model_validator(mode='after')
    def validate_pool_floor(self) -> SearchBudgetHeadTrainingConfiguration:
        if self.minimum_labelled_rows > self.batch_size:
            raise ValueError('The labelled-pool floor cannot exceed the search-budget head batch size.')
        return self


class SearchBudgetConfiguration(FrozenModel):
    curve_version: Literal['live_ema_ten_bucket_v1'] = 'live_ema_ten_bucket_v1'
    labeling: DeepLabelingConfiguration = DeepLabelingConfiguration()
    calibration: CurveCalibrationConfiguration = CurveCalibrationConfiguration()
    production: ProductionAllocationConfiguration = ProductionAllocationConfiguration()
    head_training: SearchBudgetHeadTrainingConfiguration = SearchBudgetHeadTrainingConfiguration()
