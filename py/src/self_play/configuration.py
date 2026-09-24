from __future__ import annotations

from enum import Enum
from math import isclose, isfinite, log
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, JsonValue, model_serializer, model_validator
from src.self_play.parameters import (
    ParentValueFirstPlayUrgencyParameters,
    RandomOpeningStartParameters,
    ReducedParentValueFirstPlayUrgencyParameters,
    ResolvedSelfPlayParameters,
    RestartStateStartParameters,
    ZeroFirstPlayUrgencyParameters,
)
from src.training.quantization.configuration import QatCheckpointPhase
from src.util.frozen_model import ConfigurationPath, FrozenModel
from src.util.generation_schedule import (
    FloatGenerationSchedule,
    IntegerGenerationSchedule,
    defined_schedule_values,
)


class SdpaBackend(str, Enum):
    AUTOMATIC = 'automatic'
    FLASH = 'flash'
    MEMORY_EFFICIENT = 'memory_efficient'
    MATH = 'math'
    CUDNN = 'cudnn'


class InferencePrecision(str, Enum):
    BFLOAT16 = 'bfloat16'
    FLOAT16 = 'float16'
    FLOAT32 = 'float32'


class InferenceMemoryFormat(str, Enum):
    CONTIGUOUS = 'contiguous'
    CHANNELS_LAST = 'channels_last'


class TorchScriptInferenceBackend(FrozenModel):
    kind: Literal['torchscript'] = 'torchscript'


class TensorRtFloatTemplate(FrozenModel):
    kind: Literal['float'] = 'float'
    model_id: str
    engine_path: ConfigurationPath


class TensorRtQatFloatTemplate(FrozenModel):
    kind: Literal['qat_float'] = 'qat_float'
    model_id: str
    phase: QatCheckpointPhase
    engine_path: ConfigurationPath


class TensorRtQatTemplate(FrozenModel):
    kind: Literal['qat'] = 'qat'
    model_id: str
    phase: QatCheckpointPhase
    engine_path: ConfigurationPath


TensorRtTemplate: TypeAlias = Annotated[
    TensorRtFloatTemplate | TensorRtQatFloatTemplate | TensorRtQatTemplate,
    Field(discriminator='kind'),
]


class TensorRtTemplatePrecision(str, Enum):
    FLOAT = 'float'
    INT8 = 'int8'


class TensorRtInferenceBackend(FrozenModel):
    kind: Literal['tensorrt'] = 'tensorrt'
    templates: tuple[TensorRtTemplate, ...] = Field(min_length=1)
    bootstrap_with_torchscript: bool = False
    allow_fidelity_deviation: bool = False

    @model_validator(mode='after')
    def validate_templates(self) -> TensorRtInferenceBackend:
        identities: list[tuple[str, TensorRtTemplatePrecision, QatCheckpointPhase | None]] = []
        for template in self.templates:
            match template:
                case TensorRtFloatTemplate(model_id=model_id):
                    identities.append((model_id, TensorRtTemplatePrecision.FLOAT, None))
                case TensorRtQatFloatTemplate(model_id=model_id, phase=phase):
                    identities.append((model_id, TensorRtTemplatePrecision.FLOAT, phase))
                case TensorRtQatTemplate(model_id=model_id, phase=phase):
                    identities.append((model_id, TensorRtTemplatePrecision.INT8, phase))
        if len(set(identities)) != len(identities):
            raise ValueError('TensorRT template model and phase identities must be unique.')
        return self

    def template_engine_path(
        self,
        model_id: str,
        precision: TensorRtTemplatePrecision,
        qat_phase: QatCheckpointPhase | None,
    ) -> Path:
        for template in self.templates:
            match template, precision, qat_phase:
                case TensorRtFloatTemplate(model_id=template_model_id), TensorRtTemplatePrecision.FLOAT, None if (
                    template_model_id == model_id
                ):
                    return template.engine_path
                case TensorRtQatFloatTemplate(
                    model_id=template_model_id,
                    phase=phase,
                ), TensorRtTemplatePrecision.FLOAT, candidate_phase if (
                    template_model_id == model_id and phase is candidate_phase
                ):
                    return template.engine_path
                case TensorRtQatTemplate(
                    model_id=template_model_id,
                    phase=phase,
                ), TensorRtTemplatePrecision.INT8, candidate_phase if (
                    template_model_id == model_id and phase is candidate_phase
                ):
                    return template.engine_path
        phase_name = 'unquantized' if qat_phase is None else qat_phase.value
        raise ValueError(
            f'No {precision.value} TensorRT template is configured for model {model_id} in phase {phase_name}.'
        )


InferenceBackendConfiguration: TypeAlias = Annotated[
    TorchScriptInferenceBackend | TensorRtInferenceBackend,
    Field(discriminator='kind'),
]


class DisabledForcedPlayoutConfiguration(FrozenModel):
    kind: Literal['disabled'] = 'disabled'

    def resolved_coefficient(self) -> float:
        return 0.0


class EnabledForcedPlayoutConfiguration(FrozenModel):
    kind: Literal['enabled'] = 'enabled'
    coefficient: float = Field(gt=0.0)

    @model_validator(mode='after')
    def validate_coefficient(self) -> EnabledForcedPlayoutConfiguration:
        if not isfinite(self.coefficient):
            raise ValueError('Forced-playout coefficient must be finite.')
        return self

    def resolved_coefficient(self) -> float:
        return self.coefficient


ForcedPlayoutConfiguration: TypeAlias = Annotated[
    DisabledForcedPlayoutConfiguration | EnabledForcedPlayoutConfiguration,
    Field(discriminator='kind'),
]


class ZeroFirstPlayUrgencyConfiguration(FrozenModel):
    kind: Literal['zero'] = 'zero'

    def resolve(self, model_generation: int) -> ZeroFirstPlayUrgencyParameters:
        del model_generation
        return ZeroFirstPlayUrgencyParameters()


class ParentValueFirstPlayUrgencyConfiguration(FrozenModel):
    kind: Literal['parent_value'] = 'parent_value'

    def resolve(self, model_generation: int) -> ParentValueFirstPlayUrgencyParameters:
        del model_generation
        return ParentValueFirstPlayUrgencyParameters()


class ReducedParentValueFirstPlayUrgencyConfiguration(FrozenModel):
    kind: Literal['reduced_parent_value'] = 'reduced_parent_value'
    reduction: FloatGenerationSchedule

    @model_validator(mode='after')
    def validate_reduction(self) -> ReducedParentValueFirstPlayUrgencyConfiguration:
        if any(not isfinite(value) or value <= 0.0 for value in defined_schedule_values(self.reduction)):
            raise ValueError('Reduced-parent FPU reduction must remain finite and positive.')
        return self

    def resolve(self, model_generation: int) -> ReducedParentValueFirstPlayUrgencyParameters:
        return ReducedParentValueFirstPlayUrgencyParameters(reduction=self.reduction.value_at(model_generation))


FirstPlayUrgencyConfiguration: TypeAlias = Annotated[
    ZeroFirstPlayUrgencyConfiguration
    | ParentValueFirstPlayUrgencyConfiguration
    | ReducedParentValueFirstPlayUrgencyConfiguration,
    Field(discriminator='kind'),
]


# AlphaZero scales its PUCT constant with the visit count rather than fixing it; see the pseudocode
# accompanying Silver et al. (2018).
_ALPHAZERO_EXPLORATION_BASE = 19652.0
_ALPHAZERO_EXPLORATION_INIT = 1.25


def alphazero_exploration_constant(searches_per_move: int) -> float:
    numerator = searches_per_move + _ALPHAZERO_EXPLORATION_BASE + 1.0
    return log(numerator / _ALPHAZERO_EXPLORATION_BASE) + _ALPHAZERO_EXPLORATION_INIT


class SelfPlaySearchParams(FrozenModel):
    baseline_visits: IntegerGenerationSchedule
    virtual_loss_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    dirichlet_epsilon: FloatGenerationSchedule
    dirichlet_alpha: FloatGenerationSchedule
    exploration_constant: Literal['auto'] | FloatGenerationSchedule
    first_play_urgency: FirstPlayUrgencyConfiguration
    forced_playouts: ForcedPlayoutConfiguration

    def resolved_exploration_constant(self, model_generation: int) -> float:
        # 'auto' follows the AlphaZero schedule at whatever visit budget is in force that generation,
        # which is what the evaluation path has always used.
        if self.exploration_constant == 'auto':
            return alphazero_exploration_constant(self.baseline_visits.value_at(model_generation))
        return self.exploration_constant.value_at(model_generation)

    @model_validator(mode='after')
    def validate_scheduled_values(self) -> SelfPlaySearchParams:
        if any(value <= 0 for value in defined_schedule_values(self.baseline_visits)):
            raise ValueError('Every baseline visit budget must be positive.')
        if any(not 0.0 <= value <= 1.0 for value in defined_schedule_values(self.dirichlet_epsilon)):
            raise ValueError('Dirichlet epsilon must remain in [0, 1].')
        if any(value <= 0.0 for value in defined_schedule_values(self.dirichlet_alpha)):
            raise ValueError('Dirichlet alpha must remain positive.')
        if self.exploration_constant != 'auto' and any(
            value <= 0.0 for value in defined_schedule_values(self.exploration_constant)
        ):
            raise ValueError('Exploration constant must remain positive.')
        return self


class BatchedInferenceParams(FrozenModel):
    inference_workers: int = Field(gt=0)
    inference_batch_size: int = Field(gt=0)
    outstanding_batches_per_worker: int = Field(ge=1, le=2)
    backend: InferenceBackendConfiguration = TorchScriptInferenceBackend()
    sdpa_backend: SdpaBackend = SdpaBackend.AUTOMATIC
    precision: InferencePrecision = InferencePrecision.BFLOAT16
    memory_format: InferenceMemoryFormat = InferenceMemoryFormat.CONTIGUOUS
    cudnn_benchmark: bool = False

    @model_validator(mode='before')
    @classmethod
    def preserve_existing_automatic_dispatch(
        cls,
        configuration: BatchedInferenceParams | dict[str, JsonValue],
    ) -> BatchedInferenceParams | dict[str, JsonValue]:
        match configuration:
            case dict():
                # Every omitted key resolves to the shipped path, so a configuration written before
                # these knobs existed keeps running unchanged.
                return {
                    'backend': {'kind': 'torchscript'},
                    'sdpa_backend': SdpaBackend.AUTOMATIC.value,
                    'precision': InferencePrecision.BFLOAT16.value,
                    'memory_format': InferenceMemoryFormat.CONTIGUOUS.value,
                    'cudnn_benchmark': False,
                    **configuration,
                }
            case BatchedInferenceParams():
                return configuration

    @model_serializer
    def omit_unset_execution_knobs(self) -> dict[str, JsonValue]:
        # A configuration that does not opt in must serialise, and therefore hash, exactly as it did
        # before these knobs existed, so no recorded experiment_configuration_sha256 moves.
        payload: dict[str, JsonValue] = {
            'inference_workers': self.inference_workers,
            'inference_batch_size': self.inference_batch_size,
            'outstanding_batches_per_worker': self.outstanding_batches_per_worker,
            'sdpa_backend': self.sdpa_backend.value,
        }
        if isinstance(self.backend, TensorRtInferenceBackend):
            payload['backend'] = self.backend.model_dump(mode='json')
        if not (
            self.precision is InferencePrecision.BFLOAT16
            and self.memory_format is InferenceMemoryFormat.CONTIGUOUS
            and not self.cudnn_benchmark
        ):
            payload['precision'] = self.precision.value
            payload['memory_format'] = self.memory_format.value
            payload['cudnn_benchmark'] = self.cudnn_benchmark
        return payload


class RandomOpeningStartConfiguration(FrozenModel):
    kind: Literal['random_opening'] = 'random_opening'
    maximum_plies: IntegerGenerationSchedule

    @model_validator(mode='after')
    def validate_maximum_plies(self) -> RandomOpeningStartConfiguration:
        if any(value < 0 for value in defined_schedule_values(self.maximum_plies)):
            raise ValueError('Maximum random opening plies must remain nonnegative.')
        return self

    def resolve(self, model_generation: int) -> RandomOpeningStartParameters:
        return RandomOpeningStartParameters(kind=self.kind, maximum_plies=self.maximum_plies.value_at(model_generation))


class RestartStateStartConfiguration(FrozenModel):
    kind: Literal['restart_state'] = 'restart_state'
    standard_start_probability: float = Field(ge=0.0, le=1.0)
    random_start_probability: float = Field(ge=0.0, le=1.0)
    restart_start_probability: float = Field(ge=0.0, le=1.0)
    maximum_random_opening_plies: int = Field(ge=0)
    uniform_restart_probability: float = Field(ge=0.0, le=1.0)
    candidate_visit_mass: float = Field(gt=0.0, le=1.0)
    minimum_candidates: int = Field(ge=2)
    maximum_candidates: int = Field(ge=2)
    maximum_absolute_root_value: float = Field(ge=0.0, le=1.0)
    minimum_remaining_plies: int = Field(gt=0)
    maximum_archive_positions: int = Field(gt=0)
    maximum_age_generations: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_candidate_count(self) -> RestartStateStartConfiguration:
        if self.maximum_candidates < self.minimum_candidates:
            raise ValueError('Maximum restart candidates must not be below the minimum.')
        if not isclose(
            self.standard_start_probability + self.random_start_probability + self.restart_start_probability,
            1.0,
        ):
            raise ValueError('Restart-state start probabilities must sum to one.')
        if (
            self.restart_start_probability > 0.0
            and self.standard_start_probability + self.random_start_probability == 0.0
        ):
            raise ValueError('Restart-state self-play requires a non-restart fallback.')
        return self

    def resolve(self, model_generation: int) -> RestartStateStartParameters:
        del model_generation
        return RestartStateStartParameters(
            kind=self.kind,
            standard_start_probability=self.standard_start_probability,
            random_start_probability=self.random_start_probability,
            restart_start_probability=self.restart_start_probability,
            maximum_random_opening_plies=self.maximum_random_opening_plies,
            uniform_restart_probability=self.uniform_restart_probability,
            candidate_visit_mass=self.candidate_visit_mass,
            minimum_candidates=self.minimum_candidates,
            maximum_candidates=self.maximum_candidates,
            maximum_absolute_root_value=self.maximum_absolute_root_value,
            minimum_remaining_plies=self.minimum_remaining_plies,
            maximum_archive_positions=self.maximum_archive_positions,
            maximum_age_generations=self.maximum_age_generations,
        )


StartPositionConfiguration: TypeAlias = Annotated[
    RandomOpeningStartConfiguration | RestartStateStartConfiguration,
    Field(discriminator='kind'),
]


class SelfPlayConfiguration(FrozenModel):
    search: SelfPlaySearchParams
    inference: BatchedInferenceParams
    start_position: StartPositionConfiguration
    retained_root_visit_fraction: FloatGenerationSchedule
    greedy_after_ply: IntegerGenerationSchedule
    starting_temperature: FloatGenerationSchedule
    final_temperature: FloatGenerationSchedule
    primary_sample_weight: FloatGenerationSchedule
    detailed_statistics_workers: int = Field(default=1, ge=0)

    @model_validator(mode='after')
    def validate_temperatures(self) -> SelfPlayConfiguration:
        for schedule, name in (
            (self.starting_temperature, 'Starting temperature'),
            (self.final_temperature, 'Final temperature'),
        ):
            if any(value <= 0.0 for value in defined_schedule_values(schedule)):
                raise ValueError(f'{name} must remain positive.')
        if any(value <= 0 for value in defined_schedule_values(self.greedy_after_ply)):
            raise ValueError('Greedy ply must remain positive.')
        if any(not 0.0 <= value <= 1.0 for value in defined_schedule_values(self.retained_root_visit_fraction)):
            raise ValueError('Retained-root fraction must remain in [0, 1].')
        if any(value <= 0.0 for value in defined_schedule_values(self.primary_sample_weight)):
            raise ValueError('Primary sample weight must remain positive.')
        return self

    def resolve(
        self,
        model_generation: int,
        maximum_game_plies: int | None,
        value_discount_per_ply: float,
    ) -> ResolvedSelfPlayParameters:
        search = self.search
        return ResolvedSelfPlayParameters(
            start_position=self.start_position.resolve(model_generation),
            baseline_visits=search.baseline_visits.value_at(model_generation),
            virtual_loss_weight=search.virtual_loss_weight,
            forced_playout_coefficient=search.forced_playouts.resolved_coefficient(),
            exploration_constant=search.resolved_exploration_constant(model_generation),
            first_play_urgency=search.first_play_urgency.resolve(model_generation),
            dirichlet_alpha=search.dirichlet_alpha.value_at(model_generation),
            dirichlet_epsilon=search.dirichlet_epsilon.value_at(model_generation),
            retained_root_visit_fraction=self.retained_root_visit_fraction.value_at(model_generation),
            starting_temperature=self.starting_temperature.value_at(model_generation),
            final_temperature=self.final_temperature.value_at(model_generation),
            greedy_after_ply=self.greedy_after_ply.value_at(model_generation),
            maximum_game_plies=maximum_game_plies,
            primary_sample_weight=self.primary_sample_weight.value_at(model_generation),
            value_discount_per_ply=value_discount_per_ply,
        )
