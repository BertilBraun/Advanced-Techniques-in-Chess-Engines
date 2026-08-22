import pytest
from pydantic import TypeAdapter, ValidationError

from src.util.generation_schedule import (
    ConstantSchedule,
    FloatGenerationSchedule,
    GenerationStage,
    IntegerGenerationSchedule,
    LinearSchedule,
    ScheduleRounding,
    StagedSchedule,
)


def test_constant_schedule_returns_typed_value() -> None:
    integer = ConstantSchedule[int](value=7)
    floating = ConstantSchedule[float](value=0.25)

    assert integer.value_at(100) == 7
    assert isinstance(integer.value_at(100), int)
    assert floating.value_at(100) == pytest.approx(0.25)
    assert isinstance(floating.value_at(100), float)


def test_numeric_schedule_values_wrap_and_serialize_as_constants() -> None:
    integer_adapter = TypeAdapter(IntegerGenerationSchedule)
    float_adapter = TypeAdapter(FloatGenerationSchedule)

    integer = integer_adapter.validate_python(7)
    floating = float_adapter.validate_python(0.25)

    assert integer == ConstantSchedule[int](value=7)
    assert floating == ConstantSchedule[float](value=0.25)
    assert integer_adapter.dump_python(integer, mode='json') == 7
    assert float_adapter.dump_python(floating, mode='json') == pytest.approx(0.25)
    assert integer_adapter.validate_json(integer_adapter.dump_json(integer)) == integer
    assert float_adapter.validate_json(float_adapter.dump_json(floating)) == floating


@pytest.mark.parametrize(
    ('adapter', 'payload'),
    (
        (TypeAdapter(IntegerGenerationSchedule), {'kind': 'constant', 'value': 7}),
        (TypeAdapter(FloatGenerationSchedule), {'kind': 'constant', 'value': 0.25}),
    ),
)
def test_explicit_constant_schedule_mapping_is_rejected(
    adapter: TypeAdapter[IntegerGenerationSchedule] | TypeAdapter[FloatGenerationSchedule],
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match='numeric value'):
        adapter.validate_python(payload)


def test_staged_schedule_selects_boundaries() -> None:
    schedule = StagedSchedule[int](
        stages=(
            GenerationStage[int](start_generation=0, value=8),
            GenerationStage[int](start_generation=3, value=16),
            GenerationStage[int](start_generation=9, value=32),
        )
    )

    assert tuple(schedule.value_at(generation) for generation in (0, 2, 3, 8, 9, 20)) == (8, 8, 16, 16, 32, 32)


@pytest.mark.parametrize(
    ('rounding', 'expected'),
    (
        (ScheduleRounding.FLOOR, (0, 2, 5, 7, 10)),
        (ScheduleRounding.NEAREST, (0, 3, 5, 8, 10)),
        (ScheduleRounding.CEILING, (0, 3, 5, 8, 10)),
    ),
)
def test_linear_integer_schedule_rounding_and_clamping(
    rounding: ScheduleRounding,
    expected: tuple[int, ...],
) -> None:
    schedule = LinearSchedule[int](
        start_generation=2,
        end_generation=6,
        start_value=0,
        end_value=10,
        rounding=rounding,
    )

    assert tuple(schedule.value_at(generation) for generation in (0, 3, 4, 5, 20)) == expected
    assert all(isinstance(schedule.value_at(generation), int) for generation in range(7))


def test_linear_float_schedule_interpolates_and_clamps() -> None:
    schedule = LinearSchedule[float](
        start_generation=2,
        end_generation=6,
        start_value=1.0,
        end_value=0.0,
        rounding=ScheduleRounding.NONE,
    )

    assert tuple(schedule.value_at(generation) for generation in (0, 2, 3, 4, 5, 6, 20)) == pytest.approx(
        (1.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0)
    )
    assert all(isinstance(schedule.value_at(generation), float) for generation in range(7))


def test_nearest_rounding_is_symmetric_around_zero() -> None:
    schedule = LinearSchedule[int](
        start_generation=0,
        end_generation=4,
        start_value=-5,
        end_value=5,
        rounding=ScheduleRounding.NEAREST,
    )

    assert tuple(schedule.value_at(generation) for generation in range(5)) == (-5, -3, 0, 3, 5)


@pytest.mark.parametrize(
    'payload',
    (
        {'kind': 'staged', 'stages': []},
        {'kind': 'staged', 'stages': [{'start_generation': 1, 'value': 2}]},
        {
            'kind': 'staged',
            'stages': [
                {'start_generation': 0, 'value': 2},
                {'start_generation': 0, 'value': 3},
            ],
        },
        {
            'kind': 'staged',
            'stages': [
                {'start_generation': 0, 'value': 2},
                {'start_generation': 4, 'value': 3},
                {'start_generation': 2, 'value': 4},
            ],
        },
        {
            'kind': 'linear',
            'start_generation': 4,
            'end_generation': 4,
            'start_value': 0,
            'end_value': 1,
            'rounding': 'nearest',
        },
        {
            'kind': 'linear',
            'start_generation': 0,
            'end_generation': 4,
            'start_value': 0,
            'end_value': 1,
            'rounding': 'none',
        },
    ),
)
def test_invalid_integer_schedule_definitions(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(IntegerGenerationSchedule).validate_python(payload)


def test_float_linear_schedule_rejects_integer_rounding() -> None:
    with pytest.raises(ValidationError, match='rounding mode none'):
        TypeAdapter(FloatGenerationSchedule).validate_python(
            {
                'kind': 'linear',
                'start_generation': 0,
                'end_generation': 4,
                'start_value': 0.0,
                'end_value': 1.0,
                'rounding': 'ceiling',
            }
        )


@pytest.mark.parametrize('generation', (-1, -100))
def test_schedule_evaluation_rejects_negative_generation(generation: int) -> None:
    with pytest.raises(ValueError, match='nonnegative'):
        ConstantSchedule[int](value=1).value_at(generation)
