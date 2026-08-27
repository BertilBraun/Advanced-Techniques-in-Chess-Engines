from __future__ import annotations

import pytest
from src.self_play.configuration import BatchedInferenceParams, InferenceMemoryFormat, InferencePrecision, SdpaBackend
from src.self_play.native_configuration import (
    native_execution_options,
    native_inference_memory_format,
    native_inference_precision,
    native_sdpa_backend,
)

pytestmark = pytest.mark.native

native = pytest.importorskip('AlphaZeroCpp')


def _inference_parameters(**overrides: object) -> BatchedInferenceParams:
    return BatchedInferenceParams.model_validate(
        {
            'inference_workers': 1,
            'inference_batch_size': 320,
            'outstanding_batches_per_worker': 2,
            **overrides,
        }
    )


def test_default_execution_options_match_the_shipped_path() -> None:
    options = native.InferenceExecutionOptions()
    assert options.sdpa_backend == native.SdpaBackend.AUTOMATIC
    assert options.precision == native.InferencePrecision.BFLOAT16
    assert options.memory_format == native.InferenceMemoryFormat.CONTIGUOUS
    assert options.cudnn_benchmark is False


def test_an_unconfigured_inference_block_maps_to_the_native_defaults() -> None:
    options = native_execution_options(_inference_parameters())
    default = native.InferenceExecutionOptions()
    assert options.sdpa_backend == default.sdpa_backend
    assert options.precision == default.precision
    assert options.memory_format == default.memory_format
    assert options.cudnn_benchmark == default.cudnn_benchmark


@pytest.mark.parametrize(
    ('precision', 'expected_name'),
    (
        (InferencePrecision.BFLOAT16, 'BFLOAT16'),
        (InferencePrecision.FLOAT16, 'FLOAT16'),
        (InferencePrecision.FLOAT32, 'FLOAT32'),
    ),
)
def test_every_precision_maps_to_its_native_enumerator(precision: InferencePrecision, expected_name: str) -> None:
    assert native_inference_precision(precision) == getattr(native.InferencePrecision, expected_name)


@pytest.mark.parametrize(
    ('memory_format', 'expected_name'),
    (
        (InferenceMemoryFormat.CONTIGUOUS, 'CONTIGUOUS'),
        (InferenceMemoryFormat.CHANNELS_LAST, 'CHANNELS_LAST'),
    ),
)
def test_every_memory_format_maps_to_its_native_enumerator(
    memory_format: InferenceMemoryFormat, expected_name: str
) -> None:
    assert native_inference_memory_format(memory_format) == getattr(native.InferenceMemoryFormat, expected_name)


def test_configured_options_reach_the_native_struct() -> None:
    options = native_execution_options(
        _inference_parameters(precision='float16', memory_format='channels_last', cudnn_benchmark=True)
    )
    assert options.precision == native.InferencePrecision.FLOAT16
    assert options.memory_format == native.InferenceMemoryFormat.CHANNELS_LAST
    assert options.cudnn_benchmark is True


def test_inference_configuration_carries_the_execution_options() -> None:
    configuration = native.InferenceConfiguration(
        device_id=0,
        model_path='model.jit.pt',
        execution_options=native_execution_options(_inference_parameters(precision='float32')),
    )
    assert configuration.execution_options.precision == native.InferencePrecision.FLOAT32


def test_the_sdpa_backend_property_still_reads_and_writes() -> None:
    configuration = native.InferenceConfiguration(device_id=0, model_path='model.jit.pt')
    assert configuration.sdpa_backend == native.SdpaBackend.AUTOMATIC
    configuration.sdpa_backend = native_sdpa_backend(SdpaBackend.MEMORY_EFFICIENT)
    assert configuration.execution_options.sdpa_backend == native.SdpaBackend.MEMORY_EFFICIENT
