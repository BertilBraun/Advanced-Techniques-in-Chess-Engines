from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.evaluation.configuration import EvaluationSearchConfiguration
from src.self_play.configuration import (
    BatchedInferenceParams,
    SdpaBackend,
    SelfPlaySearchParams,
    TensorRtFloatTemplate,
    TensorRtInferenceBackend,
    TensorRtQatFloatTemplate,
    TensorRtQatTemplate,
    TensorRtTemplatePrecision,
    alphazero_exploration_constant,
)
from src.training.quantization.configuration import QatCheckpointPhase


def _search_params_payload() -> dict[str, object]:
    return {
        'baseline_visits': {
            'kind': 'staged',
            'stages': [
                {'start_generation': 0, 'value': 200},
                {'start_generation': 30, 'value': 600},
            ],
        },
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,
        'exploration_constant': 1.5,
        'first_play_urgency': {'kind': 'zero'},
        'forced_playouts': {'kind': 'disabled'},
    }


def test_baseline_visits_resolve_as_generation_staged_schedule() -> None:
    search = SelfPlaySearchParams.model_validate(_search_params_payload())

    assert tuple(search.baseline_visits.value_at(generation) for generation in (0, 29, 30, 100)) == (
        200,
        200,
        600,
        600,
    )


def test_constant_baseline_visits_resolve() -> None:
    payload = _search_params_payload()
    payload['baseline_visits'] = 300
    search = SelfPlaySearchParams.model_validate(payload)

    assert search.baseline_visits.value_at(0) == 300
    assert search.baseline_visits.value_at(500) == 300


def test_baseline_visits_must_remain_positive() -> None:
    payload = _search_params_payload()
    payload['baseline_visits'] = 0

    with pytest.raises(ValidationError, match='baseline visit budget'):
        SelfPlaySearchParams.model_validate(payload)


def test_virtual_loss_weight_defaults_to_full_loss_and_rejects_out_of_range() -> None:
    search = SelfPlaySearchParams.model_validate(_search_params_payload())
    assert search.virtual_loss_weight == 1.0

    weighted_payload = _search_params_payload()
    weighted_payload['virtual_loss_weight'] = 0.5
    assert SelfPlaySearchParams.model_validate(weighted_payload).virtual_loss_weight == 0.5

    invalid_payload = _search_params_payload()
    invalid_payload['virtual_loss_weight'] = 1.5
    with pytest.raises(ValidationError):
        SelfPlaySearchParams.model_validate(invalid_payload)


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
    assert configuration.model_dump(exclude_unset=True) == configuration.model_dump()


def test_tensorrt_template_selection_is_model_and_qat_phase_specific() -> None:
    backend = TensorRtInferenceBackend(
        templates=(
            TensorRtFloatTemplate(model_id='small', engine_path='small-float.engine'),
            TensorRtQatFloatTemplate(
                model_id='small',
                phase=QatCheckpointPhase.PRE_FOLD,
                engine_path='small-pre-fold-float.engine',
            ),
            TensorRtQatFloatTemplate(
                model_id='small',
                phase=QatCheckpointPhase.DEPLOYMENT,
                engine_path='small-deployment-float.engine',
            ),
            TensorRtQatTemplate(
                model_id='small',
                phase=QatCheckpointPhase.PRE_FOLD,
                engine_path='small-pre-fold.engine',
            ),
            TensorRtQatTemplate(
                model_id='small',
                phase=QatCheckpointPhase.DEPLOYMENT,
                engine_path='small-deployment.engine',
            ),
            TensorRtQatTemplate(
                model_id='medium',
                phase=QatCheckpointPhase.DEPLOYMENT,
                engine_path='medium-deployment.engine',
            ),
        )
    )

    assert backend.template_engine_path('small', TensorRtTemplatePrecision.FLOAT, None).name == 'small-float.engine'
    assert (
        backend.template_engine_path('small', TensorRtTemplatePrecision.FLOAT, QatCheckpointPhase.PRE_FOLD).name
        == 'small-pre-fold-float.engine'
    )
    assert (
        backend.template_engine_path('small', TensorRtTemplatePrecision.FLOAT, QatCheckpointPhase.DEPLOYMENT).name
        == 'small-deployment-float.engine'
    )
    assert (
        backend.template_engine_path('small', TensorRtTemplatePrecision.INT8, QatCheckpointPhase.PRE_FOLD).name
        == 'small-pre-fold.engine'
    )
    assert (
        backend.template_engine_path('small', TensorRtTemplatePrecision.INT8, QatCheckpointPhase.DEPLOYMENT).name
        == 'small-deployment.engine'
    )
    assert (
        backend.template_engine_path('medium', TensorRtTemplatePrecision.INT8, QatCheckpointPhase.DEPLOYMENT).name
        == 'medium-deployment.engine'
    )
    assert backend.model_dump(mode='json')['templates'][0]['engine_path'] == 'small-float.engine'
    assert backend.model_dump(mode='json')['allow_fidelity_deviation'] is False
    with pytest.raises(ValueError, match='medium.*pre_fold'):
        backend.template_engine_path('medium', TensorRtTemplatePrecision.INT8, QatCheckpointPhase.PRE_FOLD)


def test_tensorrt_fidelity_deviation_can_be_nonfatal() -> None:
    backend = TensorRtInferenceBackend(
        templates=(TensorRtFloatTemplate(model_id='small', engine_path='small-float.engine'),),
        allow_fidelity_deviation=True,
    )

    assert backend.allow_fidelity_deviation
    assert backend.model_dump(mode='json')['allow_fidelity_deviation'] is True


def test_tensorrt_template_identities_must_be_unique() -> None:
    with pytest.raises(ValidationError, match='identities must be unique'):
        TensorRtInferenceBackend(
            templates=(
                TensorRtQatTemplate(
                    model_id='small',
                    phase=QatCheckpointPhase.DEPLOYMENT,
                    engine_path='first.engine',
                ),
                TensorRtQatTemplate(
                    model_id='small',
                    phase=QatCheckpointPhase.DEPLOYMENT,
                    engine_path='second.engine',
                ),
            )
        )


def test_fixed_exploration_constant_ignores_the_visit_budget() -> None:
    search = SelfPlaySearchParams.model_validate(_search_params_payload())

    assert search.resolved_exploration_constant(0) == pytest.approx(1.5)
    assert search.resolved_exploration_constant(30) == pytest.approx(1.5)


def test_automatic_exploration_constant_follows_the_visit_budget() -> None:
    payload = _search_params_payload() | {'exploration_constant': 'auto'}
    search = SelfPlaySearchParams.model_validate(payload)

    assert search.resolved_exploration_constant(0) == pytest.approx(alphazero_exploration_constant(200))
    assert search.resolved_exploration_constant(30) == pytest.approx(alphazero_exploration_constant(600))
    assert search.resolved_exploration_constant(0) < search.resolved_exploration_constant(30)


def test_automatic_exploration_constant_matches_the_evaluation_path() -> None:
    payload = _search_params_payload() | {'exploration_constant': 'auto'}
    search = SelfPlaySearchParams.model_validate(payload)

    assert search.resolved_exploration_constant(30) == pytest.approx(
        EvaluationSearchConfiguration.model_validate(
            {
                'searches_per_move': 600,
                'parallel_searches': 1,
                'exploration_constant': 'auto',
                'inference': {
                    'inference_workers': 1,
                    'inference_batch_size': 64,
                    'outstanding_batches_per_worker': 1,
                },
            }
        ).resolved_exploration_constant
    )
