from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from src.training.architecture_catalog import ArchitectureCatalogEntry, ParameterBand, load_architecture_catalog
from src.training.network import AttentionNetworkParams, Network, NetworkDefinition


CATALOG_PATH = Path('configs/architectures/chess-cnn-attention-v1.yaml')


def catalog_entries() -> tuple[ArchitectureCatalogEntry, ...]:
    return load_architecture_catalog(CATALOG_PATH).models


@pytest.mark.parametrize('entry', catalog_entries(), ids=lambda entry: entry.model_id)
def test_chess_architecture_parameter_counts(entry: ArchitectureCatalogEntry) -> None:
    definition = entry.definition
    model = Network(
        definition.architecture,
        torch.device('cpu'),
        definition.dimensions,
        definition.auxiliary_output_sizes,
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == entry.expected_training_parameters


@pytest.mark.parametrize(
    'entry',
    tuple(entry for entry in catalog_entries() if entry.parameter_band is ParameterBand.ONE_MILLION),
    ids=lambda entry: entry.model_id,
)
def test_chess_backbones_support_cpu_forward_backward_and_canonical_heads(entry: ArchitectureCatalogEntry) -> None:
    definition = entry.definition
    model = Network(
        definition.architecture,
        torch.device('cpu'),
        definition.dimensions,
        definition.auxiliary_output_sizes,
    )
    states = torch.randn(
        1,
        definition.dimensions.channels,
        definition.dimensions.rows,
        definition.dimensions.columns,
    )

    output = model.training_output(states)
    loss = (
        output.policy_logits.square().mean()
        + output.wdl_logits.square().mean()
        + sum(auxiliary.square().mean() for auxiliary in output.auxiliary_logits)
    )
    loss.backward()

    assert output.policy_logits.shape == (1, definition.dimensions.actions)
    assert output.wdl_logits.shape == (1, definition.dimensions.outcomes)
    assert tuple(tensor.shape for tensor in output.auxiliary_logits) == ((1, 1880), (1, 1))
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize('entry', catalog_entries(), ids=lambda entry: entry.model_id)
def test_chess_architecture_configuration_round_trip(entry: ArchitectureCatalogEntry) -> None:
    serialized = entry.definition.model_dump_json()

    assert NetworkDefinition.model_validate_json(serialized) == entry.definition


def test_attention_configuration_requires_even_head_partition() -> None:
    with pytest.raises(ValidationError, match='divisible'):
        AttentionNetworkParams(
            num_layers=2,
            embedding_size=63,
            num_heads=4,
            feedforward_size=128,
        )
