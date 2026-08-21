import math

import pytest
import torch

from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.training.initialization_guard import (
    InitializationProbe,
    assert_healthy_policy_initialization,
    probe_policy_initialization,
    summarize_policy_initialization,
)
from src.training.network import (
    AttentionNetworkParams,
    Chess76PlaneDirectPolicyHeadConfiguration,
    DisabledResidualContext,
    Network,
    NetworkParams,
)


def _chess_probe_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    states = (torch.rand((batch_size, 29, 8, 8), generator=generator) < 0.15).float()
    states[:, 22:] = torch.rand((batch_size, 7, 1, 1), generator=generator)
    legal_action_ids = torch.full((batch_size, 24), -1, dtype=torch.int64)
    for row in range(batch_size):
        legal_count = 12 + row % 12
        legal_action_ids[row, :legal_count] = torch.randperm(4864, generator=generator)[:legal_count]
    return states, legal_action_ids


@pytest.mark.parametrize(
    'parameters',
    (
        NetworkParams(
            num_layers=2,
            hidden_size=64,
            residual_context=DisabledResidualContext(),
            policy_head=Chess76PlaneDirectPolicyHeadConfiguration(),
        ),
        AttentionNetworkParams(
            num_layers=2,
            embedding_size=64,
            num_heads=2,
            feedforward_size=128,
            policy_head=Chess76PlaneDirectPolicyHeadConfiguration(),
        ),
    ),
)
def test_freshly_initialized_chess_networks_pass_the_startup_guard(
    parameters: NetworkParams | AttentionNetworkParams,
) -> None:
    torch.manual_seed(13)
    model = Network(parameters, torch.device('cpu'), CHESS_NETWORK_DIMENSIONS)
    states, legal_action_ids = _chess_probe_batch(16)

    probe = probe_policy_initialization(model, states, legal_action_ids)

    assert probe.policy_logit_std <= 1.5
    assert probe.policy_entropy_ratio >= 0.6
    assert_healthy_policy_initialization(probe)


def test_summary_masks_padding_and_averages_over_rows() -> None:
    policy_logits = torch.zeros((2, 8))
    policy_logits[0, :4] = torch.tensor((0.0, 0.0, 0.0, 1000.0))
    legal_action_ids = torch.tensor(((0, 1, 2, 3, -1), (4, 5, -1, -1, -1)))

    probe = summarize_policy_initialization(policy_logits, legal_action_ids)

    first_row_std = float(torch.tensor((0.0, 0.0, 0.0, 1000.0)).std(correction=0))
    assert probe.policy_logit_std == pytest.approx((first_row_std + 0.0) / 2)
    assert probe.policy_entropy_ratio == pytest.approx((0.0 + 1.0) / 2, abs=1e-6)


def test_summary_requires_a_position_with_multiple_legal_actions() -> None:
    with pytest.raises(ValueError, match='two or more legal actions'):
        summarize_policy_initialization(torch.zeros((1, 4)), torch.tensor(((2, -1, -1),)))


def test_guard_rejects_wide_logits_and_collapsed_entropy() -> None:
    with pytest.raises(RuntimeError, match='policy-logit std'):
        assert_healthy_policy_initialization(InitializationProbe(policy_logit_std=5.0, policy_entropy_ratio=0.9))
    with pytest.raises(RuntimeError, match='entropy ratio'):
        assert_healthy_policy_initialization(InitializationProbe(policy_logit_std=0.1, policy_entropy_ratio=0.2))


def test_entropy_ratio_uses_natural_log_of_legal_count() -> None:
    policy_logits = torch.zeros((1, 6))
    legal_action_ids = torch.tensor(((0, 1, 2, 3, 4, 5),))

    probe = summarize_policy_initialization(policy_logits, legal_action_ids)

    assert probe.policy_entropy_ratio == pytest.approx(math.log(6) / math.log(6))
    assert probe.policy_logit_std == pytest.approx(0.0)
