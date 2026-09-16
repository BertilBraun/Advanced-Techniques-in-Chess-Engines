from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import onnx
from pydantic import Field
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256


class RefitTensorIdentity(FrozenModel):
    name: str = Field(min_length=1)
    source: Literal['initializer', 'constant']
    onnx_data_type: int | None = Field(default=None, ge=0)
    dimensions: tuple[int, ...]


class OnnxRefitContract(FrozenModel):
    schema_version: Literal[1] = 1
    tensors: tuple[RefitTensorIdentity, ...] = Field(min_length=1)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode('utf-8')).hexdigest()


class TensorRtRefitTemplateMetadata(FrozenModel):
    schema_version: Literal[1] = 1
    engine_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    onnx_refit_contract: OnnxRefitContract


def template_metadata_path(engine_path: Path) -> Path:
    return engine_path.with_suffix(f'{engine_path.suffix}.refit.json')


def canonicalize_onnx_refit_names(model: onnx.ModelProto) -> None:
    occupied_names = {name for node in model.graph.node for name in (*node.input, *node.output) if name}
    occupied_names.update(initializer.name for initializer in model.graph.initializer if initializer.name)
    replacements: dict[str, str] = {}
    for node_index, node in enumerate(model.graph.node):
        if not node.name:
            node.name = f'node.{node_index:05d}.{node.op_type}'
        if node.op_type != 'Constant':
            continue
        for output_index, output_name in enumerate(node.output):
            if not output_name:
                raise ValueError('Every ONNX Constant output must have a name.')
            canonical_name = f'refit.constant.{node_index:05d}.{output_index}'
            if canonical_name in occupied_names and canonical_name != output_name:
                raise ValueError(f'Canonical TensorRT refit name is already occupied: {canonical_name}')
            replacements[output_name] = canonical_name
            occupied_names.add(canonical_name)

    if not replacements:
        return
    for node in model.graph.node:
        for input_index, input_name in enumerate(node.input):
            node.input[input_index] = replacements.get(input_name, input_name)
        for output_index, output_name in enumerate(node.output):
            node.output[output_index] = replacements.get(output_name, output_name)
    for graph_value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        graph_value.name = replacements.get(graph_value.name, graph_value.name)


def onnx_refit_contract(model: onnx.ModelProto) -> OnnxRefitContract:
    identities: list[RefitTensorIdentity] = []
    for initializer in model.graph.initializer:
        if not initializer.name:
            raise ValueError('Every ONNX initializer must have a stable name.')
        identities.append(
            RefitTensorIdentity(
                name=initializer.name,
                source='initializer',
                onnx_data_type=initializer.data_type,
                dimensions=tuple(initializer.dims),
            )
        )
    for node in model.graph.node:
        if node.op_type != 'Constant':
            continue
        if len(node.output) != 1:
            raise ValueError('Every TensorRT-refittable ONNX Constant must have exactly one output.')
        value = next((attribute.t for attribute in node.attribute if attribute.name == 'value'), None)
        identities.append(
            RefitTensorIdentity(
                name=node.output[0],
                source='constant',
                onnx_data_type=None if value is None else value.data_type,
                dimensions=() if value is None else tuple(value.dims),
            )
        )
    identities.sort(key=lambda identity: (identity.source, identity.name))
    names = tuple(identity.name for identity in identities)
    if len(set(names)) != len(names):
        raise ValueError('TensorRT refit tensor names must be unique across initializers and constants.')
    return OnnxRefitContract(tensors=tuple(identities))


def load_template_metadata(engine_path: Path) -> TensorRtRefitTemplateMetadata:
    path = template_metadata_path(engine_path)
    if not path.is_file():
        raise ValueError(f'TensorRT template is missing its refit contract: {path}')
    metadata = TensorRtRefitTemplateMetadata.model_validate_json(path.read_text(encoding='utf-8'))
    if metadata.engine_sha256 != file_sha256(engine_path):
        raise ValueError(f'TensorRT template refit contract does not match the engine: {engine_path}')
    return metadata
