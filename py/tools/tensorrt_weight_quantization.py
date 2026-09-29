from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

WEIGHT_PRODUCT_OPERATOR_TYPES = frozenset({'MatMul', 'Gemm', 'Conv'})
# Exported weights can reach a product through these without mixing in an activation.
WEIGHT_PASSTHROUGH_OPERATOR_TYPES = frozenset({'Transpose', 'Cast', 'Identity', 'Reshape', 'Unsqueeze', 'Squeeze'})
HEAD_EXCLUDED_NODE_PATTERNS = (r'.*policy_head.*', r'.*value_head.*')


class LowPrecision(str, Enum):
    FP8 = 'fp8'
    INT8 = 'int8'


@dataclass(frozen=True)
class GraphNode:
    name: str
    operator_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


def weight_derived_tensors(nodes: Iterable[GraphNode], initializer_names: Iterable[str]) -> frozenset[str]:
    derived = set(initializer_names)
    for node in nodes:
        if node.operator_type == 'Constant':
            derived.update(node.outputs)
        elif node.operator_type in WEIGHT_PASSTHROUGH_OPERATOR_TYPES and all(
            name in derived for name in node.inputs if name
        ):
            derived.update(node.outputs)
    return frozenset(derived)


def weight_product_node_patterns(
    nodes: tuple[GraphNode, ...],
    initializer_names: Iterable[str],
    excluded_node_patterns: tuple[str, ...] = HEAD_EXCLUDED_NODE_PATTERNS,
) -> tuple[str, ...]:
    """Anchored name patterns for every product of an activation with a stored weight.

    Attention's activation-by-activation products stay in float16: Q/DQ there breaks TensorRT's fused
    attention kernel and costs more than the lower precision saves.
    """
    weights = weight_derived_tensors(nodes, initializer_names)
    excluded = tuple(re.compile(pattern) for pattern in excluded_node_patterns)
    return tuple(
        f'^{re.escape(node.name)}$'
        for node in nodes
        if node.operator_type in WEIGHT_PRODUCT_OPERATOR_TYPES
        and len(node.inputs) > 1
        and node.inputs[0] not in weights
        and node.inputs[1] in weights
        and not any(pattern.fullmatch(node.name) for pattern in excluded)
    )


def require_precision_support(precision: LowPrecision, compute_capability: tuple[int, int]) -> None:
    if precision is LowPrecision.FP8 and compute_capability < (8, 9):
        raise ValueError(
            f'FP8 needs compute capability 8.9 (Ada) or newer; this GPU is {compute_capability[0]}.'
            f'{compute_capability[1]}.'
        )
