from __future__ import annotations

import pytest
from tools.tensorrt_weight_quantization import (
    GraphNode,
    LowPrecision,
    require_precision_support,
    weight_derived_tensors,
    weight_product_node_patterns,
)


def _attention_block() -> tuple[GraphNode, ...]:
    return (
        GraphNode('/encoder.0/query/Transpose', 'Transpose', ('query.weight',), ('query_weight_t',)),
        GraphNode('/encoder.0/query/MatMul', 'MatMul', ('tokens', 'query_weight_t'), ('queries',)),
        GraphNode('/encoder.0/key/MatMul', 'MatMul', ('tokens', 'key.weight'), ('keys',)),
        GraphNode('/encoder.0/scores/MatMul', 'MatMul', ('queries', 'keys'), ('scores',)),
        GraphNode('/encoder.0/Softmax', 'Softmax', ('scores',), ('weights',)),
        GraphNode('/encoder.0/context/MatMul', 'MatMul', ('weights', 'values'), ('context',)),
        GraphNode('/policy_head/MatMul', 'MatMul', ('context', 'policy.weight'), ('policy',)),
        GraphNode('/value_head/Gemm', 'Gemm', ('context', 'value.weight', 'value.bias'), ('value',)),
    )


INITIALIZERS = ('query.weight', 'key.weight', 'policy.weight', 'value.weight', 'value.bias')


def test_transposed_weight_counts_as_weight_derived() -> None:
    assert 'query_weight_t' in weight_derived_tensors(_attention_block(), INITIALIZERS)


def test_passthrough_of_an_activation_is_not_weight_derived() -> None:
    nodes = (GraphNode('reshape', 'Reshape', ('tokens', 'shape'), ('reshaped',)),)
    assert 'reshaped' not in weight_derived_tensors(nodes, ('shape',))


def test_constant_output_counts_as_weight_derived() -> None:
    nodes = (GraphNode('template', 'Constant', (), ('template_bank',)),)
    assert 'template_bank' in weight_derived_tensors(nodes, ())


def test_selection_keeps_only_activation_by_weight_products_outside_the_heads() -> None:
    assert weight_product_node_patterns(_attention_block(), INITIALIZERS) == (
        r'^/encoder\.0/query/MatMul$',
        r'^/encoder\.0/key/MatMul$',
    )


def test_selection_includes_heads_when_nothing_is_excluded() -> None:
    patterns = weight_product_node_patterns(_attention_block(), INITIALIZERS, excluded_node_patterns=())
    assert patterns[-2:] == (r'^/policy_head/MatMul$', r'^/value_head/Gemm$')


@pytest.mark.parametrize('compute_capability', [(8, 6), (8, 0), (7, 5)])
def test_fp8_is_refused_before_ada(compute_capability: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match='FP8'):
        require_precision_support(LowPrecision.FP8, compute_capability)


@pytest.mark.parametrize(
    ('precision', 'compute_capability'),
    [(LowPrecision.FP8, (8, 9)), (LowPrecision.FP8, (9, 0)), (LowPrecision.INT8, (8, 6))],
)
def test_supported_precision_is_accepted(precision: LowPrecision, compute_capability: tuple[int, int]) -> None:
    require_precision_support(precision, compute_capability)
