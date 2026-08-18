from src.self_play.configuration import BatchedInferenceParams, SdpaBackend


def test_batched_inference_backend_is_typed_and_serialized() -> None:
    configuration = BatchedInferenceParams(
        inference_workers=2,
        inference_batch_size=64,
        outstanding_batches_per_worker=2,
        sdpa_backend=SdpaBackend.MEMORY_EFFICIENT,
    )

    assert configuration.sdpa_backend is SdpaBackend.MEMORY_EFFICIENT
    assert configuration.model_dump(mode='json')['sdpa_backend'] == 'memory_efficient'


def test_existing_inference_configuration_preserves_automatic_dispatch() -> None:
    configuration = BatchedInferenceParams(
        inference_workers=1,
        inference_batch_size=64,
        outstanding_batches_per_worker=1,
    )

    assert configuration.sdpa_backend is SdpaBackend.AUTOMATIC
