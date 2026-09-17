from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from tools.publish_tensorrt_engine import _policy_distribution_agreement, onnx_graph_signature


def _write_constant_graph(path: Path, value: np.ndarray, consumer: str) -> None:
    constant = helper.make_node('Constant', (), ('constant',), value=numpy_helper.from_array(value), name='constant')
    if consumer == 'QuantizeLinear':
        zero = helper.make_tensor('zero', TensorProto.INT8, (), (0,))
        node = helper.make_node(consumer, ('input', 'constant', 'zero'), ('output',))
        output_type = TensorProto.INT8
    elif consumer == 'Reshape':
        node = helper.make_node(consumer, ('input', 'constant'), ('output',))
        output_type = TensorProto.FLOAT
    else:
        raise ValueError(f'Unsupported test consumer: {consumer}')
    graph = helper.make_graph(
        (constant, node),
        'constant-signature',
        (helper.make_tensor_value_info('input', TensorProto.FLOAT, None),),
        (helper.make_tensor_value_info('output', output_type, None),),
        (() if consumer == 'Reshape' else (zero,)),
    )
    onnx.save(helper.make_model(graph), path)


def test_policy_distribution_agreement_is_exact_for_equal_logits() -> None:
    logits = np.array(((1.0, 2.0, 3.0), (3.0, 2.0, 1.0)), dtype=np.float32)

    top1_agreement, mean_divergence, maximum_divergence = _policy_distribution_agreement(logits, logits)

    assert top1_agreement == 1.0
    assert mean_divergence == pytest.approx(0.0, abs=1e-7)
    assert maximum_divergence == pytest.approx(0.0, abs=1e-7)


def test_policy_distribution_agreement_detects_changed_policy() -> None:
    reference = np.array(((5.0, 0.0, -1.0), (0.0, 5.0, -1.0)), dtype=np.float32)
    candidate = np.array(((0.0, 5.0, -1.0), (0.0, 5.0, -1.0)), dtype=np.float32)

    top1_agreement, mean_divergence, maximum_divergence = _policy_distribution_agreement(reference, candidate)

    assert top1_agreement == 0.5
    assert mean_divergence > 0.0
    assert maximum_divergence > mean_divergence


def test_graph_signature_ignores_quantization_parameter_values(tmp_path: Path) -> None:
    first = tmp_path / 'first.onnx'
    second = tmp_path / 'second.onnx'
    _write_constant_graph(first, np.array(0.125, dtype=np.float32), 'QuantizeLinear')
    _write_constant_graph(second, np.array(0.25, dtype=np.float32), 'QuantizeLinear')

    assert onnx_graph_signature(first) == onnx_graph_signature(second)


def test_graph_signature_preserves_structural_constant_values(tmp_path: Path) -> None:
    first = tmp_path / 'first.onnx'
    second = tmp_path / 'second.onnx'
    _write_constant_graph(first, np.array((1, 4), dtype=np.int64), 'Reshape')
    _write_constant_graph(second, np.array((2, 2), dtype=np.int64), 'Reshape')

    assert onnx_graph_signature(first) != onnx_graph_signature(second)
