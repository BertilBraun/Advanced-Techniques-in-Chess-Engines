from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt
import torch
from AlphaZeroCpp import (
    InferenceBackend,
    InferenceDevice,
    InferenceDimensions,
    InferenceExecutionOptions,
    InferenceMemoryFormat,
    InferencePrecision,
    InferenceRunner,
    SdpaBackend,
)
from src.util.atomic_file import write_bytes_atomically, write_text_atomically
from src.util.hashing import file_sha256

INPUT_NAME = 'states'
POLICY_OUTPUT_NAME = 'policy_logits'
WDL_OUTPUT_NAME = 'wdl_probabilities'
ONNX_OPSET_VERSION = 18
MAXIMUM_OUTPUT_MEAN_ABSOLUTE_ERROR = 0.01
MAXIMUM_WDL_ABSOLUTE_ERROR = 0.1
MINIMUM_POLICY_TOP1_AGREEMENT = 0.90
# A correctly calibrated INT8 export of the 14x160 measures 0.00155 mean and 0.00626 maximum policy
# KL against its float reference, and a miscalibrated one measures 0.42 and 2.06. The old 1e-3 and
# 0.01 sat below the healthy case, so every export failed and the warning said nothing.
MAXIMUM_POLICY_MEAN_KL_DIVERGENCE = 1e-2
MAXIMUM_POLICY_MAXIMUM_KL_DIVERGENCE = 0.05


@dataclass(frozen=True)
class TensorRtVerification:
    batch_size: int
    policy_mean_absolute_error: float
    policy_maximum_absolute_error: float
    policy_top1_agreement: float
    policy_mean_kl_divergence: float
    policy_maximum_kl_divergence: float
    wdl_mean_absolute_error: float
    wdl_maximum_absolute_error: float
    fidelity_limits_passed: bool


def onnx_graph_signature(path: Path) -> str:
    model = onnx.load(path, load_external_data=False)
    consumers: dict[str, list[tuple[str, int]]] = {}
    for node in model.graph.node:
        for input_index, input_name in enumerate(node.input):
            consumers.setdefault(input_name, []).append((node.op_type, input_index))

    def attribute_signature(attribute: onnx.AttributeProto, normalize_tensor_value: bool) -> tuple[str, int, str]:
        if normalize_tensor_value and attribute.type == onnx.AttributeProto.TENSOR:
            tensor_shape = (attribute.t.data_type, tuple(attribute.t.dims))
            return attribute.name, attribute.type, json.dumps(tensor_shape, separators=(',', ':'))
        return attribute.name, attribute.type, attribute.SerializeToString().hex()

    def is_quantization_parameter_constant(node: onnx.NodeProto) -> bool:
        if node.op_type != 'Constant' or not node.output:
            return False
        uses = tuple(use for output_name in node.output for use in consumers.get(output_name, ()))
        return bool(uses) and all(
            consumer_type in ('QuantizeLinear', 'DequantizeLinear') and input_index in (1, 2)
            for consumer_type, input_index in uses
        )

    initializers = tuple(
        (initializer.name, initializer.data_type, tuple(initializer.dims))
        for initializer in sorted(model.graph.initializer, key=lambda item: item.name)
    )
    nodes = tuple(
        (
            node.domain,
            node.op_type,
            tuple(node.input),
            tuple(node.output),
            tuple(
                attribute_signature(attribute, is_quantization_parameter_constant(node))
                for attribute in sorted(node.attribute, key=lambda item: item.name)
            ),
        )
        for node in model.graph.node
    )
    graph_inputs = tuple((value.name, value.type.SerializeToString().hex()) for value in model.graph.input)
    graph_outputs = tuple((value.name, value.type.SerializeToString().hex()) for value in model.graph.output)
    opsets = tuple(sorted((opset.domain, opset.version) for opset in model.opset_import))
    payload = json.dumps(
        (opsets, graph_inputs, graph_outputs, nodes, initializers),
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def engine_input_shape(engine: trt.ICudaEngine) -> tuple[int, int, int, int]:
    shape = engine.get_tensor_shape(INPUT_NAME)
    if len(shape) != 4 or any(dimension <= 0 for dimension in shape):
        raise ValueError(f'TensorRT template has incompatible input shape: {tuple(shape)}')
    return tuple(shape)


def export_onnx_with_example(model_path: Path, output_path: Path, example: torch.Tensor) -> None:
    model = torch.jit.load(str(model_path), map_location='cpu').to(dtype=torch.float16).eval()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (example,),
            str(output_path),
            input_names=(INPUT_NAME,),
            output_names=(POLICY_OUTPUT_NAME, WDL_OUTPUT_NAME),
            opset_version=ONNX_OPSET_VERSION,
            do_constant_folding=False,
            dynamo=False,
        )
    exported = onnx.load(output_path)
    onnx.checker.check_model(exported, full_check=True)


def export_onnx(model_path: Path, output_path: Path, input_shape: tuple[int, int, int, int]) -> None:
    export_onnx_with_example(model_path, output_path, torch.zeros(input_shape, dtype=torch.float16))


def refit_engine(template_path: Path, onnx_path: Path, output_path: Path) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(template_path.read_bytes())
    if engine is None:
        raise ValueError(f'Could not deserialize TensorRT template: {template_path}')
    refitter = trt.Refitter(engine, logger)
    parser_refitter = trt.OnnxParserRefitter(refitter, logger)
    if not parser_refitter.refit_from_file(str(onnx_path)):
        raise ValueError(f'TensorRT could not refit from {onnx_path}')
    missing = tuple(sorted(refitter.get_missing_weights()))
    if missing:
        raise ValueError(f'TensorRT refit is missing weights: {missing}')
    if not refitter.refit_cuda_engine():
        raise ValueError('TensorRT engine refit failed')
    write_bytes_atomically(output_path, bytes(engine.serialize()))


def _onnx_output_width(model: onnx.ModelProto, name: str) -> int:
    matching = tuple(output for output in model.graph.output if output.name == name)
    if len(matching) != 1:
        raise ValueError(f'ONNX graph must have exactly one output named {name}.')
    dimensions = matching[0].type.tensor_type.shape.dim
    if len(dimensions) != 2 or not dimensions[1].HasField('dim_value') or dimensions[1].dim_value <= 0:
        raise ValueError(f'ONNX output {name} must have a static positive width.')
    return dimensions[1].dim_value


def _onnx_outputs(onnx_path: Path, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    providers: list[str | tuple[str, dict[str, str]]] = [
        ('CUDAExecutionProvider', {'device_id': '0'}),
        'CPUExecutionProvider',
    ]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_metadata = session.get_inputs()
    if len(input_metadata) != 1 or input_metadata[0].name != INPUT_NAME:
        raise ValueError('The deployed ONNX graph must have exactly one input named states.')
    match input_metadata[0].type:
        case 'tensor(float)':
            onnx_states = states.astype(np.float32)
        case 'tensor(float16)':
            onnx_states = states.astype(np.float16)
        case input_type:
            raise ValueError(f'Unsupported deployed ONNX input type: {input_type}.')
    policy_logits, wdl_probabilities = session.run(
        (POLICY_OUTPUT_NAME, WDL_OUTPUT_NAME),
        {INPUT_NAME: onnx_states},
    )
    return policy_logits.astype(np.float32), wdl_probabilities.astype(np.float32)


def _output_errors(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f'TensorRT output shape {candidate.shape} does not match ONNX shape {reference.shape}.')
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError('ONNX and TensorRT verification outputs must be finite.')
    errors = np.abs(reference - candidate)
    return float(errors.mean()), float(errors.max())


def _policy_distribution_agreement(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f'TensorRT policy shape {candidate.shape} does not match ONNX shape {reference.shape}.')
    reference_shifted = reference - reference.max(axis=1, keepdims=True)
    candidate_shifted = candidate - candidate.max(axis=1, keepdims=True)
    reference_log_probabilities = reference_shifted - np.log(np.exp(reference_shifted).sum(axis=1, keepdims=True))
    candidate_log_probabilities = candidate_shifted - np.log(np.exp(candidate_shifted).sum(axis=1, keepdims=True))
    reference_probabilities = np.exp(reference_log_probabilities)
    divergences = np.maximum(
        np.sum(reference_probabilities * (reference_log_probabilities - candidate_log_probabilities), axis=1),
        0.0,
    )
    top1_agreement = np.mean(reference.argmax(axis=1) == candidate.argmax(axis=1))
    return float(top1_agreement), float(divergences.mean()), float(divergences.max())


def verify_engine(
    onnx_path: Path,
    engine_path: Path,
    input_shape: tuple[int, int, int, int],
    allow_fidelity_deviation: bool,
) -> TensorRtVerification:
    batch_size = input_shape[0]
    generator = np.random.default_rng(0)
    states = generator.integers(0, 2, size=(batch_size, *input_shape[1:]), dtype=np.int8)
    onnx_policy, onnx_wdl = _onnx_outputs(onnx_path, states)
    model = onnx.load(onnx_path, load_external_data=False)
    dimensions = InferenceDimensions(
        input_shape[1],
        input_shape[2],
        input_shape[3],
        _onnx_output_width(model, POLICY_OUTPUT_NAME),
        _onnx_output_width(model, WDL_OUTPUT_NAME),
    )
    runner = InferenceRunner(
        model_path=str(engine_path),
        device=InferenceDevice.CUDA,
        device_id=0,
        maximum_batch_size=input_shape[0],
        use_dedicated_cuda_stream=True,
        dimensions=dimensions,
        execution_options=InferenceExecutionOptions(
            sdpa_backend=SdpaBackend.AUTOMATIC,
            precision=InferencePrecision.FLOAT16,
            memory_format=InferenceMemoryFormat.CONTIGUOUS,
            cudnn_benchmark=False,
        ),
        backend=InferenceBackend.TENSORRT,
    )
    tensor_rt_policy, tensor_rt_wdl = runner.forward(states)
    policy_mean_error, policy_maximum_error = _output_errors(onnx_policy, tensor_rt_policy)
    policy_top1_agreement, policy_mean_kl_divergence, policy_maximum_kl_divergence = _policy_distribution_agreement(
        onnx_policy, tensor_rt_policy
    )
    wdl_mean_error, wdl_maximum_error = _output_errors(onnx_wdl, tensor_rt_wdl)
    fidelity_limits_passed = not (
        policy_top1_agreement < MINIMUM_POLICY_TOP1_AGREEMENT
        or policy_mean_kl_divergence > MAXIMUM_POLICY_MEAN_KL_DIVERGENCE
        or policy_maximum_kl_divergence > MAXIMUM_POLICY_MAXIMUM_KL_DIVERGENCE
        or wdl_mean_error > MAXIMUM_OUTPUT_MEAN_ABSOLUTE_ERROR
        or wdl_maximum_error > MAXIMUM_WDL_ABSOLUTE_ERROR
    )
    if not fidelity_limits_passed and not allow_fidelity_deviation:
        raise ValueError(
            'TensorRT verification failed: '
            f'policy mean/max={policy_mean_error:.6f}/{policy_maximum_error:.6f}, '
            f'policy top1/KL mean/max={policy_top1_agreement:.6f}/'
            f'{policy_mean_kl_divergence:.6f}/{policy_maximum_kl_divergence:.6f}, '
            f'WDL mean/max={wdl_mean_error:.6f}/{wdl_maximum_error:.6f}.'
        )
    return TensorRtVerification(
        batch_size=batch_size,
        policy_mean_absolute_error=policy_mean_error,
        policy_maximum_absolute_error=policy_maximum_error,
        policy_top1_agreement=policy_top1_agreement,
        policy_mean_kl_divergence=policy_mean_kl_divergence,
        policy_maximum_kl_divergence=policy_maximum_kl_divergence,
        wdl_mean_absolute_error=wdl_mean_error,
        wdl_maximum_absolute_error=wdl_maximum_error,
        fidelity_limits_passed=fidelity_limits_passed,
    )


def _automatic_template_path(
    configured_template_path: Path, input_shape: tuple[int, int, int, int], signature: str
) -> Path:
    return configured_template_path.parent / 'automatic-refit' / f'b{input_shape[0]}-{signature}.engine'


def _build_automatic_template(
    onnx_path: Path,
    configured_template_path: Path,
    input_shape: tuple[int, int, int, int],
    signature: str,
) -> Path:
    template_path = _automatic_template_path(configured_template_path, input_shape, signature)
    lock_path = template_path.with_suffix('.lock')
    with exclusive_lock(lock_path):
        if template_path.is_file():
            return template_path
        template_path.parent.mkdir(parents=True, exist_ok=True)
        command = (
            sys.executable,
            '-m',
            'tools.build_tensorrt_refit_template',
            '--model',
            str(onnx_path),
            '--output',
            str(template_path),
            '--batch-size',
            str(input_shape[0]),
            '--channels',
            str(input_shape[1]),
            '--rows',
            str(input_shape[2]),
            '--columns',
            str(input_shape[3]),
            '--refit-mode',
            'all',
        )
        subprocess.run(command, check=True, cwd=Path(__file__).parents[1])
    return template_path


def publish(
    model_path: Path,
    template_paths: tuple[Path, ...],
    allow_fidelity_deviation: bool,
) -> dict[str, str | int | float | bool]:
    if not template_paths:
        raise ValueError('At least one TensorRT template is required.')
    model_path = model_path.resolve()
    template_paths = tuple(path.resolve() for path in template_paths)
    source_sha256 = file_sha256(model_path)
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    configured_template = runtime.deserialize_cuda_engine(template_paths[0].read_bytes())
    if configured_template is None:
        raise ValueError(f'Could not deserialize TensorRT template: {template_paths[0]}')
    input_shape = engine_input_shape(configured_template)
    temporary_onnx_path = model_path.with_suffix('.publication.temporary.onnx')
    owns_onnx = model_path.suffix != '.onnx'
    onnx_path = temporary_onnx_path if owns_onnx else model_path
    if owns_onnx:
        temporary_onnx_path.unlink(missing_ok=True)
        export_onnx(model_path, temporary_onnx_path, input_shape)
    try:
        exported = onnx.load(onnx_path)
        onnx.checker.check_model(exported, full_check=True)
        graph_signature = onnx_graph_signature(onnx_path)
        selected_template_path = _build_automatic_template(
            onnx_path,
            template_paths[0],
            input_shape,
            graph_signature,
        )
        template_sha256 = file_sha256(selected_template_path)
        engine_path = model_path.with_suffix(f'.trt-{template_sha256[:16]}.engine')
        metadata_path = engine_path.with_suffix('.json')
        lock_path = engine_path.with_suffix('.lock')
        with exclusive_lock(lock_path):
            if engine_path.is_file() and metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                if (
                    metadata.get('source_sha256') == source_sha256
                    and metadata.get('graph_signature') == graph_signature
                    and metadata.get('template_sha256') == template_sha256
                    and metadata.get('engine_sha256') == file_sha256(engine_path)
                    and metadata.get('verification_batch_size') == input_shape[0]
                    and (allow_fidelity_deviation or metadata.get('fidelity_limits_passed', True))
                ):
                    return {
                        **metadata,
                        'engine_path': str(engine_path),
                        'template_path': str(selected_template_path),
                        'cached': True,
                    }
            refit_started_at = time.perf_counter()
            refit_engine(selected_template_path, onnx_path, engine_path)
            refit_seconds = time.perf_counter() - refit_started_at
            verification = verify_engine(onnx_path, engine_path, input_shape, allow_fidelity_deviation)
            metadata = {
                'engine_path': str(engine_path),
                'engine_sha256': file_sha256(engine_path),
                'source_sha256': source_sha256,
                'graph_signature': graph_signature,
                'template_sha256': template_sha256,
                'template_path': str(selected_template_path),
                'batch_size': input_shape[0],
                'refit_seconds': refit_seconds,
                'verification_batch_size': verification.batch_size,
                'policy_mean_absolute_error': verification.policy_mean_absolute_error,
                'policy_maximum_absolute_error': verification.policy_maximum_absolute_error,
                'policy_top1_agreement': verification.policy_top1_agreement,
                'policy_mean_kl_divergence': verification.policy_mean_kl_divergence,
                'policy_maximum_kl_divergence': verification.policy_maximum_kl_divergence,
                'wdl_mean_absolute_error': verification.wdl_mean_absolute_error,
                'wdl_maximum_absolute_error': verification.wdl_maximum_absolute_error,
                'fidelity_limits_passed': verification.fidelity_limits_passed,
                'cached': False,
            }
            write_text_atomically(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + '\n')
            return metadata
    finally:
        if owns_onnx:
            temporary_onnx_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Atomically refit a TensorRT template for one checkpoint.')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--template-engine', type=Path, required=True, action='append')
    parser.add_argument('--allow-fidelity-deviation', action='store_true')
    arguments = parser.parse_args()
    print(
        json.dumps(
            publish(
                arguments.model,
                tuple(arguments.template_engine),
                arguments.allow_fidelity_deviation,
            ),
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()
