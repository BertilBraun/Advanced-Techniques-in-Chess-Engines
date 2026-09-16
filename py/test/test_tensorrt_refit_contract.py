from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from src.self_play.tensorrt_refit import (
    TensorRtRefitTemplateMetadata,
    canonicalize_onnx_refit_names,
    load_template_metadata,
    onnx_refit_contract,
    template_metadata_path,
)
from src.util.hashing import file_sha256


def _model(constant_name: str, weight_shape: tuple[int, ...] = (2, 2)) -> onnx.ModelProto:
    weight = numpy_helper.from_array(np.ones(weight_shape, dtype=np.float32), name='backbone.0.weight')
    constant = helper.make_node(
        'Constant',
        (),
        (constant_name,),
        value=numpy_helper.from_array(np.array((1, 2), dtype=np.int64)),
    )
    graph = helper.make_graph(
        (constant,),
        'refit-contract',
        (helper.make_tensor_value_info('states', TensorProto.FLOAT, (1, 2)),),
        (helper.make_tensor_value_info(constant_name, TensorProto.INT64, (2,)),),
        initializer=(weight,),
    )
    return helper.make_model(graph, opset_imports=(helper.make_opsetid('', 20),))


def test_refit_contract_ignores_non_refittable_generated_constants() -> None:
    first = _model('onnx::Shape_17')
    second = _model('onnx::Shape_91')

    canonicalize_onnx_refit_names(first)
    canonicalize_onnx_refit_names(second)

    assert first.graph.node[0].output[0] == 'refit.constant.00000.0'
    assert onnx_refit_contract(first) == onnx_refit_contract(second)


def test_refit_contract_rejects_a_different_weight_shape() -> None:
    first = _model('constant')
    second = _model('constant', (4, 2))
    canonicalize_onnx_refit_names(first)
    canonicalize_onnx_refit_names(second)

    assert onnx_refit_contract(first) != onnx_refit_contract(second)


def test_template_metadata_is_bound_to_the_engine(tmp_path: Path) -> None:
    engine_path = tmp_path / 'template.engine'
    engine_path.write_bytes(b'engine')
    model = _model('constant')
    canonicalize_onnx_refit_names(model)
    metadata = TensorRtRefitTemplateMetadata(
        engine_sha256=file_sha256(engine_path),
        onnx_refit_contract=onnx_refit_contract(model),
    )
    template_metadata_path(engine_path).write_text(metadata.model_dump_json(), encoding='utf-8')

    assert load_template_metadata(engine_path) == metadata
    engine_path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='does not match'):
        load_template_metadata(engine_path)
