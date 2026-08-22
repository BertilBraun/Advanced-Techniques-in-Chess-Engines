from __future__ import annotations

import math
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Generic, Literal, TypeAlias, TypeVar

from pydantic import BeforeValidator, Field, model_serializer, model_validator

from src.util.frozen_model import FrozenModel, JsonValue


ScheduleValueT = TypeVar('ScheduleValueT')
NumericScheduleValueT = TypeVar('NumericScheduleValueT', int, float)


def _validate_generation(model_generation: int) -> None:
    if model_generation < 0:
        raise ValueError('Model generation must be nonnegative.')


class ConstantSchedule(FrozenModel, Generic[ScheduleValueT]):
    kind: Literal['constant'] = 'constant'
    value: ScheduleValueT

    def value_at(self, model_generation: int) -> ScheduleValueT:
        _validate_generation(model_generation)
        return self.value

    @model_serializer
    def serialize_constant(self) -> ScheduleValueT:
        return self.value


class GenerationStage(FrozenModel, Generic[ScheduleValueT]):
    start_generation: int = Field(ge=0)
    value: ScheduleValueT


class StagedSchedule(FrozenModel, Generic[ScheduleValueT]):
    kind: Literal['staged'] = 'staged'
    stages: tuple[GenerationStage[ScheduleValueT], ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_stages(self) -> StagedSchedule[ScheduleValueT]:
        starts = tuple(stage.start_generation for stage in self.stages)
        if starts[0] != 0:
            raise ValueError('A staged schedule must start at generation zero.')
        if starts != tuple(sorted(set(starts))):
            raise ValueError('Schedule stage generations must increase uniquely.')
        return self

    def value_at(self, model_generation: int) -> ScheduleValueT:
        _validate_generation(model_generation)
        selected = self.stages[0]
        for stage in self.stages[1:]:
            if stage.start_generation > model_generation:
                break
            selected = stage
        return selected.value


class ScheduleRounding(str, Enum):
    NONE = 'none'
    FLOOR = 'floor'
    NEAREST = 'nearest'
    CEILING = 'ceiling'


class LinearSchedule(FrozenModel, Generic[NumericScheduleValueT]):
    kind: Literal['linear'] = 'linear'
    start_generation: int = Field(ge=0)
    end_generation: int = Field(ge=0)
    start_value: NumericScheduleValueT
    end_value: NumericScheduleValueT
    rounding: ScheduleRounding

    @model_validator(mode='after')
    def validate_interpolation(self) -> LinearSchedule[NumericScheduleValueT]:
        if self.end_generation <= self.start_generation:
            raise ValueError('Linear schedule end generation must be greater than its start generation.')
        integer_values = isinstance(self.start_value, int) and isinstance(self.end_value, int)
        if integer_values and self.rounding is ScheduleRounding.NONE:
            raise ValueError('Integer linear schedules require floor, nearest, or ceiling rounding.')
        if not integer_values and self.rounding is not ScheduleRounding.NONE:
            raise ValueError('Float linear schedules require rounding mode none.')
        return self

    def value_at(self, model_generation: int) -> NumericScheduleValueT:
        _validate_generation(model_generation)
        if model_generation <= self.start_generation:
            return self.start_value
        if model_generation >= self.end_generation:
            return self.end_value
        progress = (model_generation - self.start_generation) / (self.end_generation - self.start_generation)
        interpolated = float(self.start_value) + (float(self.end_value) - float(self.start_value)) * progress
        match self.rounding:
            case ScheduleRounding.NONE:
                return interpolated
            case ScheduleRounding.FLOOR:
                return math.floor(interpolated)
            case ScheduleRounding.NEAREST:
                return math.floor(interpolated + 0.5) if interpolated >= 0.0 else math.ceil(interpolated - 0.5)
            case ScheduleRounding.CEILING:
                return math.ceil(interpolated)


FloatScheduleModel: TypeAlias = ConstantSchedule[float] | StagedSchedule[float] | LinearSchedule[float]
IntegerScheduleModel: TypeAlias = ConstantSchedule[int] | StagedSchedule[int] | LinearSchedule[int]


def _wrap_float_constant(value: JsonValue | FloatScheduleModel) -> JsonValue | FloatScheduleModel:
    match value:
        case bool():
            return value
        case int() | float():
            return ConstantSchedule[float](value=value)
        case Mapping() if value.get('kind') == 'constant':
            raise ValueError('Write a constant float schedule as its numeric value.')
        case _:
            return value


def _wrap_integer_constant(value: JsonValue | IntegerScheduleModel) -> JsonValue | IntegerScheduleModel:
    match value:
        case bool():
            return value
        case int():
            return ConstantSchedule[int](value=value)
        case Mapping() if value.get('kind') == 'constant':
            raise ValueError('Write a constant integer schedule as its numeric value.')
        case _:
            return value


FloatGenerationSchedule: TypeAlias = Annotated[
    Annotated[FloatScheduleModel, Field(discriminator='kind')],
    BeforeValidator(_wrap_float_constant),
]
IntegerGenerationSchedule: TypeAlias = Annotated[
    Annotated[IntegerScheduleModel, Field(discriminator='kind')],
    BeforeValidator(_wrap_integer_constant),
]


def defined_schedule_values(
    schedule: ConstantSchedule[ScheduleValueT]
    | StagedSchedule[ScheduleValueT]
    | LinearSchedule[int]
    | LinearSchedule[float],
) -> tuple[ScheduleValueT | int | float, ...]:
    match schedule:
        case ConstantSchedule(value=value):
            return (value,)
        case StagedSchedule(stages=stages):
            return tuple(stage.value for stage in stages)
        case LinearSchedule(start_value=start_value, end_value=end_value):
            return (start_value, end_value)


def schedule_change_generations(
    schedule: ConstantSchedule[ScheduleValueT]
    | StagedSchedule[ScheduleValueT]
    | LinearSchedule[int]
    | LinearSchedule[float],
) -> tuple[int, ...]:
    match schedule:
        case ConstantSchedule():
            return (0,)
        case StagedSchedule(stages=stages):
            return tuple(stage.start_generation for stage in stages)
        case LinearSchedule(start_generation=start_generation, end_generation=end_generation):
            return (0, *range(start_generation, end_generation + 1))
