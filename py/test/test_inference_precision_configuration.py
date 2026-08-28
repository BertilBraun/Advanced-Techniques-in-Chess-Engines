from __future__ import annotations

import pytest
import torch
from src.self_play.configuration import (
    BatchedInferenceParams,
    InferenceMemoryFormat,
    InferencePrecision,
    SdpaBackend,
)
from tools.measure_inference_precision_agreement import (
    MemoryFormat,
    ModelOutputs,
    Precision,
    masked_log_softmax,
    measure_agreement,
)

pytestmark = pytest.mark.filterwarnings('ignore::DeprecationWarning')


def _inference_parameters(**overrides: object) -> BatchedInferenceParams:
    return BatchedInferenceParams.model_validate(
        {
            'inference_workers': 1,
            'inference_batch_size': 320,
            'outstanding_batches_per_worker': 2,
            **overrides,
        }
    )


def test_omitted_precision_keys_resolve_to_the_shipped_path() -> None:
    parameters = _inference_parameters()
    assert parameters.precision is InferencePrecision.BFLOAT16
    assert parameters.memory_format is InferenceMemoryFormat.CONTIGUOUS
    assert parameters.cudnn_benchmark is False


def test_omitted_sdpa_backend_still_resolves_to_automatic() -> None:
    assert _inference_parameters().sdpa_backend is SdpaBackend.AUTOMATIC


@pytest.mark.parametrize(
    ('key', 'value', 'expected'),
    (
        ('precision', 'float16', InferencePrecision.FLOAT16),
        ('precision', 'float32', InferencePrecision.FLOAT32),
        ('memory_format', 'channels_last', InferenceMemoryFormat.CHANNELS_LAST),
        ('cudnn_benchmark', True, True),
    ),
)
def test_explicit_precision_keys_are_carried(key: str, value: object, expected: object) -> None:
    assert getattr(_inference_parameters(**{key: value}), key) == expected


@pytest.mark.parametrize('key', ('precision', 'memory_format'))
def test_unknown_precision_values_are_rejected(key: str) -> None:
    with pytest.raises(ValueError):
        _inference_parameters(**{key: 'int4'})


def test_explicit_sdpa_backend_survives_the_new_defaults() -> None:
    assert _inference_parameters(sdpa_backend='memory_efficient').sdpa_backend is SdpaBackend.MEMORY_EFFICIENT


def test_an_unset_inference_block_serialises_without_the_new_keys() -> None:
    # Any recorded experiment_configuration_sha256 must survive these knobs being added.
    payload = _inference_parameters().model_dump(mode='json')
    assert 'precision' not in payload
    assert 'memory_format' not in payload
    assert 'cudnn_benchmark' not in payload
    assert payload['sdpa_backend'] == 'automatic'


@pytest.mark.parametrize(
    ('key', 'value'),
    (('precision', 'float16'), ('memory_format', 'channels_last'), ('cudnn_benchmark', True)),
)
def test_opting_in_makes_every_knob_visible_to_the_hash(key: str, value: object) -> None:
    payload = _inference_parameters(**{key: value}).model_dump(mode='json')
    assert {'precision', 'memory_format', 'cudnn_benchmark'} <= payload.keys()
    assert payload[key] == value


def test_a_serialised_inference_block_round_trips() -> None:
    original = _inference_parameters(precision='float32', memory_format='channels_last')
    assert BatchedInferenceParams.model_validate(original.model_dump(mode='json')) == original


def _outputs(policy_logits: list[list[float]], wins: list[float], corrections: list[float]) -> ModelOutputs:
    win = torch.tensor(wins, dtype=torch.float32)
    return ModelOutputs(
        policy_logits=torch.tensor(policy_logits, dtype=torch.float32),
        wdl_probabilities=torch.stack((win, torch.zeros_like(win), 1.0 - win), dim=1),
        search_budget_logits=torch.tensor(corrections, dtype=torch.float32),
    )


def test_identical_outputs_agree_exactly() -> None:
    outputs = _outputs([[1.0, 2.0, 0.5]], [0.7], [0.25])
    legal_mask = torch.tensor([[True, True, True]])
    agreement = measure_agreement(
        outputs,
        outputs,
        legal_mask,
        Precision.BFLOAT16,
        MemoryFormat.CONTIGUOUS,
        cudnn_benchmark=True,
    )
    assert agreement.cudnn_benchmark
    assert agreement.legal_top1_agreement == 1.0
    assert agreement.mean_policy_kl_divergence == pytest.approx(0.0, abs=1e-12)
    assert agreement.value_mean_absolute_error == pytest.approx(0.0, abs=1e-12)
    assert agreement.search_budget_logit_mean_absolute_error == pytest.approx(0.0, abs=1e-12)


def test_illegal_actions_cannot_win_the_top1_comparison() -> None:
    # The candidate's largest logit sits on an illegal action, which the search never sees.
    reference = _outputs([[1.0, 2.0, 0.0]], [0.5], [0.1])
    candidate = _outputs([[1.0, 2.0, 99.0]], [0.5], [0.1])
    legal_mask = torch.tensor([[True, True, False]])
    agreement = measure_agreement(reference, candidate, legal_mask, Precision.FLOAT16, MemoryFormat.CONTIGUOUS)
    assert agreement.legal_top1_agreement == 1.0
    assert agreement.unrestricted_top1_agreement == 0.0


def test_a_flipped_leading_move_is_counted_as_disagreement() -> None:
    reference = _outputs([[3.0, 1.0]], [0.5], [0.1])
    candidate = _outputs([[1.0, 3.0]], [0.5], [0.1])
    legal_mask = torch.tensor([[True, True]])
    agreement = measure_agreement(reference, candidate, legal_mask, Precision.FLOAT16, MemoryFormat.CONTIGUOUS)
    assert agreement.legal_top1_agreement == 0.0
    assert agreement.mean_policy_kl_divergence > 0.0


def test_value_error_uses_the_win_minus_loss_expectation() -> None:
    reference = _outputs([[1.0, 1.0]], [0.9], [0.1])
    candidate = _outputs([[1.0, 1.0]], [0.6], [0.1])
    legal_mask = torch.tensor([[True, True]])
    agreement = measure_agreement(reference, candidate, legal_mask, Precision.FLOAT32, MemoryFormat.CONTIGUOUS)
    # Expected value is win - loss, so a 0.3 shift in win moves the expectation by 0.6.
    assert agreement.value_mean_absolute_error == pytest.approx(0.6, abs=1e-6)


def test_masked_log_softmax_gives_illegal_actions_no_mass() -> None:
    probabilities = masked_log_softmax(torch.tensor([[5.0, 1.0, 1.0]]), torch.tensor([[False, True, True]])).exp()
    assert probabilities[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-9)


def test_kl_divergence_ignores_actions_the_reference_never_plays() -> None:
    # A reference probability of zero contributes nothing, even against a wildly different candidate.
    reference = _outputs([[50.0, -50.0]], [0.5], [0.1])
    candidate = _outputs([[50.0, 0.0]], [0.5], [0.1])
    legal_mask = torch.tensor([[True, True]])
    agreement = measure_agreement(reference, candidate, legal_mask, Precision.BFLOAT16, MemoryFormat.CHANNELS_LAST)
    assert agreement.mean_policy_kl_divergence == pytest.approx(0.0, abs=1e-9)
