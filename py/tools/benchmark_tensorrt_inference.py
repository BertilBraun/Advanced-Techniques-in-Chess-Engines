"""Benchmark the production chess checkpoint against TensorRT FP16 and calibrated INT8.

This is an offline feasibility probe. It does not add a TensorRT production backend. Run it only on
an idle target GPU because TensorRT engine building performs tactic profiling and INT8 calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
import onnx
import tensorrt as trt
import torch
from modelopt.onnx.quantization import quantize
from pydantic import Field
from src.evaluation.contracts import EVALUATION_DATASET_MANIFEST_ADAPTER
from src.evaluation.dataset import dataset_manifest_path
from src.experiment.configuration import experiment_configuration_sha256, load_chess_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS, CHESS_STATE_CONTRACT
from src.replay.batch_loader import decode_states
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.training.checkpoint.contracts import load_checkpoint_manifest_path
from src.training.targets import build_training_target_layout
from src.util.atomic_file import write_bytes_atomically, write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from src.util.provenance import SourceRevision, read_source_revision
from tools.measure_inference_precision_agreement import load_positions
from tools.tensorrt_benchmark_metrics import (
    FidelityLimits,
    FidelityMetrics,
    ModelOutputs,
    TimingDistribution,
    fidelity_failures,
    measure_fidelity,
    summarize_timings,
    validate_fidelity,
)
from tools.tensorrt_calibration_sampling import EncodedStates, select_disjoint_replay_calibration
from tools.tensorrt_quantization_recovery import (
    DirectQuantizationOutput,
    QuantizationOutputIdentity,
    identify_modelopt_autotune_recovery,
)
from torch import Tensor, nn

BATCH_SIZE = 320
INPUT_NAME = 'states'
POLICY_OUTPUT_NAME = 'policy_logits'
WDL_OUTPUT_NAME = 'wdl_probabilities'
ONNX_OPSET_VERSION = 20
TENSORRT_WORKSPACE_BYTES = 4 * 1024**3
TENSORRT_BUILDER_OPTIMIZATION_LEVEL = 3
DEFAULT_CALIBRATION_POSITION_COUNT = 32_000
DEFAULT_CALIBRATION_RANDOM_SEED = 20_260_912
INT8_QUANTIZED_OPERATOR_TYPES = ('Conv',)
INT8_QUANTIZED_NODE_PATTERNS = (r'/backbone\.13/Conv$',)
INT8_EXCLUDED_NODE_PATTERNS = (r'.*policy_head.*', r'.*value_head.*')


class Backend(str, Enum):
    TORCHSCRIPT_BFLOAT16 = 'torchscript_bfloat16_channels_last'
    TENSORRT_FLOAT16 = 'tensorrt_float16'
    TENSORRT_INT8 = 'tensorrt_int8_calibrated'


class CalibrationMethod(str, Enum):
    MAX = 'max'
    ENTROPY = 'entropy'


@dataclass(frozen=True)
class EvaluationDatasetCalibrationSource:
    path: Path


@dataclass(frozen=True)
class ReplayCalibrationSource:
    path: Path
    random_seed: int


CalibrationSource = EvaluationDatasetCalibrationSource | ReplayCalibrationSource


@dataclass(frozen=True)
class BenchmarkArguments:
    configuration_path: Path
    checkpoint_manifest_path: Path
    checkpoint_generation: int
    benchmark_dataset_path: Path
    calibration_source: CalibrationSource
    artifact_directory: Path
    output_path: Path
    gpu_id: int
    warmup_iterations: int
    repetitions: int
    iterations_per_repetition: int
    calibration_position_count: int
    calibration_method: CalibrationMethod
    quantized_node_patterns: tuple[str, ...]
    autotune: bool
    fidelity_position_offset: int
    fidelity_position_count: int
    fidelity_limits: FidelityLimits
    acknowledge_gpu_load: bool


class ArtifactIdentity(FrozenModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class DatasetIdentity(FrozenModel):
    data: ArtifactIdentity
    manifest: ArtifactIdentity
    available_positions: int = Field(gt=0)
    packed_payload_bytes: int = Field(gt=0)
    representation_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    selected_positions: int = Field(gt=0)
    selected_states_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    legal_action_mask_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class EvaluationDatasetCalibrationSourceIdentity(FrozenModel):
    kind: Literal['evaluation_dataset'] = 'evaluation_dataset'
    dataset: DatasetIdentity


class ReplayCalibrationSourceIdentity(FrozenModel):
    kind: Literal['replay'] = 'replay'
    path: str = Field(min_length=1)
    file_size_bytes: int = Field(gt=0)
    header_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    layout_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    available_positions: int = Field(gt=0)
    selected_positions: int = Field(gt=0)
    random_seed: int = Field(ge=0)
    excluded_fidelity_overlap_positions: int = Field(ge=0)
    selected_logical_indices_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    selected_packed_states_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    selected_states_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


CalibrationSourceIdentity = Annotated[
    EvaluationDatasetCalibrationSourceIdentity | ReplayCalibrationSourceIdentity,
    Field(discriminator='kind'),
]


class CalibrationIdentity(FrozenModel):
    source: CalibrationSourceIdentity
    requested_positions: int = Field(gt=0)
    submitted_batches: int = Field(gt=0)
    submitted_positions: int = Field(gt=0)
    benchmark_input_overlap_positions: Literal[0] = 0
    algorithm: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    quantized_operator_types: tuple[Literal['Conv'], ...] = INT8_QUANTIZED_OPERATOR_TYPES
    quantized_node_patterns: tuple[str, ...]
    autotune: bool
    excluded_node_patterns: tuple[str, ...] = INT8_EXCLUDED_NODE_PATTERNS


class HardwareDescription(FrozenModel):
    gpu_id: int = Field(ge=0)
    device_name: str = Field(min_length=1)
    compute_capability: tuple[int, int]
    driver_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    torch_cuda_version: str = Field(min_length=1)
    onnx_version: str = Field(min_length=1)
    tensorrt_version: str = Field(min_length=1)
    tensorrt_fast_float16: bool
    tensorrt_bfloat16_builder_flag: bool
    tensorrt_fast_int8: bool
    selected_floating_precision: Literal['float16'] = 'float16'
    floating_precision_reason: str = Field(min_length=1)


class ReferenceMeasurement(FrozenModel):
    backend: Literal[Backend.TORCHSCRIPT_BFLOAT16] = Backend.TORCHSCRIPT_BFLOAT16
    timing: TimingDistribution


class CandidateMeasurement(FrozenModel):
    backend: Backend
    engine: ArtifactIdentity
    timing: TimingDistribution
    fidelity: FidelityMetrics
    fidelity_failures: tuple[str, ...]


class TensorRtInferenceBenchmarkReport(FrozenModel):
    schema_version: Literal[10] = 10
    source_revision: SourceRevision
    experiment_configuration_path: str = Field(min_length=1)
    experiment_configuration_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    checkpoint_generation: int = Field(ge=0)
    model_id: str = Field(min_length=1)
    checkpoint_manifest: ArtifactIdentity
    inference_model: ArtifactIdentity
    benchmark_dataset: DatasetIdentity
    fidelity_position_offset: int = Field(ge=0)
    fidelity_position_count: int = Field(gt=0, le=BATCH_SIZE)
    calibration: CalibrationIdentity
    input_shape: tuple[int, int, int, int]
    input_dtype: Literal['int8_source_planes'] = 'int8_source_planes'
    warmup_iterations: int = Field(ge=0)
    hardware: HardwareDescription
    float16_onnx_model: ArtifactIdentity
    int8_source_onnx_model: ArtifactIdentity
    int8_qdq_onnx_model: ArtifactIdentity
    quantization_output: QuantizationOutputIdentity
    int8_quantize_linear_node_count: int = Field(gt=0)
    int8_dequantize_linear_node_count: int = Field(gt=0)
    onnx_opset_version: int = Field(gt=0)
    tensorrt_workspace_bytes: int = Field(gt=0)
    tensorrt_builder_optimization_level: int = Field(ge=0)
    fidelity_limits: FidelityLimits
    reference: ReferenceMeasurement
    candidates: tuple[CandidateMeasurement, ...] = Field(min_length=2, max_length=2)


@dataclass(frozen=True)
class _ResolvedCheckpoint:
    model_path: Path
    model_sha256: str
    manifest_sha256: str
    model_id: str


class _CudaGraphRunner:
    def replay(self) -> None:
        raise NotImplementedError

    def outputs(self) -> ModelOutputs:
        raise NotImplementedError


class _TorchScriptCudaGraphRunner(_CudaGraphRunner):
    def __init__(self, model_path: Path, encoded_states: Tensor, device: torch.device, warmup_iterations: int) -> None:
        torch.backends.cudnn.benchmark = True
        model = torch.jit.load(str(model_path), map_location=device)
        model.to(dtype=torch.bfloat16, memory_format=torch.channels_last)
        model.eval()
        self._model: nn.Module = torch.jit.freeze(model)
        self._encoded_states = encoded_states.to(device=device, dtype=torch.int8)
        self._typed_states = torch.empty(
            encoded_states.shape,
            device=device,
            dtype=torch.bfloat16,
            memory_format=torch.channels_last,
        )
        self._policy = torch.empty((BATCH_SIZE, CHESS_NETWORK_DIMENSIONS.actions), device=device, dtype=torch.float32)
        self._wdl = torch.empty((BATCH_SIZE, 3), device=device, dtype=torch.float32)
        self._stream = torch.cuda.Stream(device=device)
        with torch.inference_mode(), torch.cuda.stream(self._stream):
            for _ in range(warmup_iterations):
                self._execute()
        self._stream.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self._graph, stream=self._stream):
            self._execute()

    def _execute(self) -> None:
        self._typed_states.copy_(self._encoded_states)
        policy, wdl = self._model(self._typed_states)
        self._policy.copy_(policy)
        self._wdl.copy_(wdl)

    def replay(self) -> None:
        self._graph.replay()

    def outputs(self) -> ModelOutputs:
        self.replay()
        torch.cuda.synchronize(self._policy.device)
        return ModelOutputs(policy_logits=self._policy.cpu(), wdl_probabilities=self._wdl.cpu())


class _TensorRtCudaGraphRunner(_CudaGraphRunner):
    def __init__(self, engine_path: Path, encoded_states: Tensor, device: torch.device, warmup_iterations: int) -> None:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise ValueError(f'TensorRT could not deserialize {engine_path}.')
        self._runtime = runtime
        self._engine: trt.ICudaEngine = engine
        context = engine.create_execution_context()
        if context is None:
            raise ValueError(f'TensorRT could not create an execution context for {engine_path}.')
        self._context: trt.IExecutionContext = context
        self._encoded_states = encoded_states.to(device=device, dtype=torch.int8)
        self._input_dtype = self._require_engine_contract()
        self._typed_states = torch.empty(encoded_states.shape, device=device, dtype=self._input_dtype)
        self._policy_engine = self._allocate_output(POLICY_OUTPUT_NAME, device)
        self._wdl_engine = self._allocate_output(WDL_OUTPUT_NAME, device)
        self._policy = torch.empty_like(self._policy_engine, dtype=torch.float32)
        self._wdl = torch.empty_like(self._wdl_engine, dtype=torch.float32)
        if not self._context.set_tensor_address(INPUT_NAME, self._typed_states.data_ptr()):
            raise ValueError('TensorRT rejected the input tensor address.')
        if not self._context.set_tensor_address(POLICY_OUTPUT_NAME, self._policy_engine.data_ptr()):
            raise ValueError('TensorRT rejected the policy output tensor address.')
        if not self._context.set_tensor_address(WDL_OUTPUT_NAME, self._wdl_engine.data_ptr()):
            raise ValueError('TensorRT rejected the WDL output tensor address.')
        self._stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(self._stream):
            for _ in range(warmup_iterations):
                self._execute()
        self._stream.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, stream=self._stream):
            self._execute()

    def _require_engine_contract(self) -> torch.dtype:
        names = tuple(self._engine.get_tensor_name(index) for index in range(self._engine.num_io_tensors))
        expected_names = (INPUT_NAME, POLICY_OUTPUT_NAME, WDL_OUTPUT_NAME)
        if set(names) != set(expected_names):
            raise ValueError(f'TensorRT engine tensors are {names}, expected {expected_names}.')
        expected_shapes = (
            (INPUT_NAME, _input_shape()),
            (POLICY_OUTPUT_NAME, (BATCH_SIZE, CHESS_NETWORK_DIMENSIONS.actions)),
            (WDL_OUTPUT_NAME, (BATCH_SIZE, 3)),
        )
        for name, expected_shape in expected_shapes:
            observed_shape = tuple(self._engine.get_tensor_shape(name))
            if observed_shape != expected_shape:
                raise ValueError(f'TensorRT tensor {name} has shape {observed_shape}, expected {expected_shape}.')
        input_data_type = self._engine.get_tensor_dtype(INPUT_NAME)
        match input_data_type:
            case trt.DataType.FLOAT:
                return torch.float32
            case trt.DataType.HALF:
                return torch.float16
            case _:
                raise ValueError(f'TensorRT benchmark engine input has unsupported data type {input_data_type}.')

    def _allocate_output(self, name: str, device: torch.device) -> Tensor:
        shape = tuple(self._engine.get_tensor_shape(name))
        data_type = self._engine.get_tensor_dtype(name)
        match data_type:
            case trt.DataType.FLOAT:
                torch_dtype = torch.float32
            case trt.DataType.HALF:
                torch_dtype = torch.float16
            case trt.DataType.BF16:
                torch_dtype = torch.bfloat16
            case _:
                raise ValueError(f'TensorRT output {name} has unsupported data type {data_type}.')
        return torch.empty(shape, device=device, dtype=torch_dtype)

    def _execute(self) -> None:
        self._typed_states.copy_(self._encoded_states)
        if not self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise ValueError('TensorRT inference submission failed.')
        self._policy.copy_(self._policy_engine)
        self._wdl.copy_(self._wdl_engine)

    def replay(self) -> None:
        self._graph.replay()

    def outputs(self) -> ModelOutputs:
        self.replay()
        torch.cuda.synchronize(self._policy.device)
        return ModelOutputs(policy_logits=self._policy.cpu(), wdl_probabilities=self._wdl.cpu())


def _tensor_sha256(tensor: Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _array_sha256(array: npt.NDArray[np.generic]) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _input_overlap_count(left: Tensor, right: Tensor) -> int:
    right_rows = {hashlib.sha256(row.numpy().tobytes()).digest() for row in right.detach().cpu().contiguous()}
    return sum(hashlib.sha256(row.numpy().tobytes()).digest() in right_rows for row in left.detach().cpu().contiguous())


def _load_dataset_identity(path: Path, selected_states: Tensor, legal_action_mask: Tensor) -> DatasetIdentity:
    manifest_path = dataset_manifest_path(path)
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(f'Benchmark dataset and manifest must both exist at {path}.')
    manifest = EVALUATION_DATASET_MANIFEST_ADAPTER.validate_json(manifest_path.read_text(encoding='utf-8'))
    if file_sha256(path) != manifest.data_sha256:
        raise ValueError(f'Dataset hash does not match its manifest: {path}')
    expected_payload_bytes = CHESS_STATE_CONTRACT.packed_plane_layout.payload_bytes
    if manifest.packed_payload_bytes != expected_payload_bytes:
        raise ValueError(
            f'Dataset packed payload is {manifest.packed_payload_bytes} bytes; '
            f'the current chess state contract requires {expected_payload_bytes}.'
        )
    return DatasetIdentity(
        data=ArtifactIdentity(path=str(path), sha256=manifest.data_sha256),
        manifest=ArtifactIdentity(path=str(manifest_path), sha256=file_sha256(manifest_path)),
        available_positions=manifest.position_count,
        packed_payload_bytes=manifest.packed_payload_bytes,
        representation_digest=manifest.representation_digest,
        selected_positions=selected_states.shape[0],
        selected_states_sha256=_tensor_sha256(selected_states),
        legal_action_mask_sha256=_tensor_sha256(legal_action_mask),
    )


def _replay_layout(configuration: ChessExperimentConfiguration) -> ReplayLayout:
    return ReplayLayout(
        packed_planes=CHESS_STATE_CONTRACT.packed_plane_layout,
        targets=build_training_target_layout(
            CHESS_NETWORK_DIMENSIONS.actions,
            configuration.chess.objective.auxiliary_targets,
        ),
        maximum_policy_entries=configuration.training.lifecycle.replay.maximum_policy_entries,
        maximum_legal_actions=CHESS_STATE_CONTRACT.maximum_legal_action_count,
    )


def _load_replay_calibration(
    source: ReplayCalibrationSource,
    position_count: int,
    configuration: ChessExperimentConfiguration,
    benchmark_states: Tensor,
) -> tuple[Tensor, ReplayCalibrationSourceIdentity]:
    if not source.path.is_file():
        raise ValueError(f'Calibration replay does not exist: {source.path}')
    layout = _replay_layout(configuration)
    store = ReplayStore.open(source.path, layout, writable=False)
    try:
        available_positions = store.state.size
        if position_count > available_positions:
            raise ValueError(
                f'Calibration requests {position_count} replay positions, but only {available_positions} exist.'
            )

        def load_replay_states(indices: npt.NDArray[np.int64]) -> tuple[EncodedStates, npt.NDArray[np.int8]]:
            encoded = store.gather_logical(indices).encoded_state.copy()
            decoded = decode_states(encoded, CHESS_STATE_CONTRACT).astype(np.int8)
            return encoded, decoded

        sample = select_disjoint_replay_calibration(
            available_positions,
            position_count,
            source.random_seed,
            benchmark_states.numpy(),
            load_replay_states,
        )
    finally:
        store.close()
    states = torch.from_numpy(sample.decoded_states)
    with source.path.open('rb') as replay_file:
        header_sha256 = hashlib.sha256(replay_file.read(65_536)).hexdigest()
    return states, ReplayCalibrationSourceIdentity(
        path=str(source.path),
        file_size_bytes=source.path.stat().st_size,
        header_sha256=header_sha256,
        layout_sha256=layout.digest,
        available_positions=available_positions,
        selected_positions=position_count,
        random_seed=source.random_seed,
        excluded_fidelity_overlap_positions=sample.excluded_overlap_count,
        selected_logical_indices_sha256=_array_sha256(sample.logical_indices),
        selected_packed_states_sha256=_array_sha256(sample.encoded_states),
        selected_states_sha256=_tensor_sha256(states),
    )


def _load_calibration(
    source: CalibrationSource,
    position_count: int,
    configuration: ChessExperimentConfiguration,
    benchmark_states: Tensor,
) -> tuple[Tensor, CalibrationSourceIdentity]:
    match source:
        case EvaluationDatasetCalibrationSource(path=path):
            states, legal_action_mask = load_positions(path, position_count)
            states = states.to(torch.int8)
            return states, EvaluationDatasetCalibrationSourceIdentity(
                dataset=_load_dataset_identity(path, states, legal_action_mask)
            )
        case ReplayCalibrationSource():
            return _load_replay_calibration(source, position_count, configuration, benchmark_states)


def _resolve_checkpoint(arguments: BenchmarkArguments) -> _ResolvedCheckpoint:
    manifest = load_checkpoint_manifest_path(arguments.checkpoint_manifest_path, arguments.checkpoint_generation)
    if manifest.network.dimensions != CHESS_NETWORK_DIMENSIONS:
        raise ValueError('Checkpoint network dimensions do not match the current chess representation.')
    configuration = load_chess_experiment_configuration(arguments.configuration_path)
    matching_models = tuple(
        model.model_id
        for model in configuration.training.progressive_model_sizing.models
        if model.network == manifest.network.architecture
    )
    if len(matching_models) != 1:
        raise ValueError('Checkpoint architecture does not identify exactly one model in the experiment configuration.')
    return _ResolvedCheckpoint(
        model_path=arguments.checkpoint_manifest_path.parent / manifest.inference_model_path,
        model_sha256=manifest.inference_model_sha256,
        manifest_sha256=file_sha256(arguments.checkpoint_manifest_path),
        model_id=matching_models[0],
    )


def _export_onnx(model_path: Path, output_path: Path, data_type: torch.dtype) -> ArtifactIdentity:
    model = torch.jit.load(str(model_path), map_location='cpu')
    model.to(dtype=data_type)
    model.eval()
    example = torch.zeros(
        _input_shape(),
        dtype=data_type,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f'.{output_path.name}.tmp')
    temporary_path.unlink(missing_ok=True)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                model,
                (example,),
                str(temporary_path),
                input_names=(INPUT_NAME,),
                output_names=(POLICY_OUTPUT_NAME, WDL_OUTPUT_NAME),
                opset_version=ONNX_OPSET_VERSION,
                do_constant_folding=True,
                dynamo=False,
            )
        exported = onnx.load(temporary_path)
        onnx.checker.check_model(exported, full_check=True)
        write_bytes_atomically(output_path, temporary_path.read_bytes())
    except Exception as error:
        raise ValueError(f'ONNX export or validation failed for {model_path}: {error}') from error
    finally:
        temporary_path.unlink(missing_ok=True)
    return ArtifactIdentity(path=str(output_path), sha256=file_sha256(output_path))


def _quantize_onnx(
    source_path: Path,
    output_path: Path,
    calibration_states: Tensor,
    device: torch.device,
    calibration_method: CalibrationMethod,
    quantized_node_patterns: tuple[str, ...],
    autotune: bool,
) -> tuple[ArtifactIdentity, int, int, QuantizationOutputIdentity]:
    output_path.unlink(missing_ok=True)
    assert device.index is not None
    autotune_output_path = output_path.parent / 'autotune'
    optimized_autotune_path = autotune_output_path / 'optimized_final.onnx'
    if autotune:
        optimized_autotune_path.unlink(missing_ok=True)
    quantization_output: QuantizationOutputIdentity = DirectQuantizationOutput()
    try:
        quantize(
            onnx_path=str(source_path),
            quantize_mode='int8',
            calibration_data=calibration_states.to(torch.float32).numpy(),
            calibration_method=calibration_method.value,
            calibration_eps=[f'cuda:{device.index}', 'cpu'],
            op_types_to_quantize=list(INT8_QUANTIZED_OPERATOR_TYPES),
            nodes_to_quantize=list(quantized_node_patterns),
            nodes_to_exclude=list(INT8_EXCLUDED_NODE_PATTERNS),
            high_precision_dtype='fp16',
            autotune=autotune,
            autotune_output_dir=str(autotune_output_path) if autotune else None,
            autotune_num_schemes_per_region=20,
            autotune_warmup_runs=10,
            autotune_timing_runs=20,
            output_path=str(output_path),
        )
    except Exception as error:
        recovery = identify_modelopt_autotune_recovery(error)
        if not autotune or recovery is None or not optimized_autotune_path.exists():
            raise ValueError(f'ModelOpt explicit INT8 Q/DQ conversion failed for {source_path}: {error}') from error
        write_bytes_atomically(output_path, optimized_autotune_path.read_bytes())
        quantization_output = recovery
    quantized_model = onnx.load(output_path)
    onnx.checker.check_model(quantized_model, full_check=True)
    quantize_linear_count = sum(node.op_type == 'QuantizeLinear' for node in quantized_model.graph.node)
    dequantize_linear_count = sum(node.op_type == 'DequantizeLinear' for node in quantized_model.graph.node)
    if quantize_linear_count == 0 or dequantize_linear_count == 0:
        raise ValueError('ModelOpt output does not contain explicit INT8 Q/DQ nodes.')
    return (
        ArtifactIdentity(path=str(output_path), sha256=file_sha256(output_path)),
        quantize_linear_count,
        dequantize_linear_count,
        quantization_output,
    )


def _parse_onnx(network: trt.INetworkDefinition, parser: trt.OnnxParser, path: Path) -> None:
    if parser.parse_from_file(str(path)):
        return
    errors = tuple(str(parser.get_error(index)) for index in range(parser.num_errors))
    raise ValueError(f'TensorRT ONNX conversion failed for {path}: {" | ".join(errors)}')


def _build_engine(
    onnx_path: Path,
    output_path: Path,
    backend: Backend,
) -> tuple[ArtifactIdentity, bool, bool]:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)
    _parse_onnx(network, parser, onnx_path)
    configuration = builder.create_builder_config()
    configuration.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, TENSORRT_WORKSPACE_BYTES)
    configuration.builder_optimization_level = TENSORRT_BUILDER_OPTIMIZATION_LEVEL
    configuration.set_flag(trt.BuilderFlag.FP16)
    match backend:
        case Backend.TENSORRT_FLOAT16:
            pass
        case Backend.TENSORRT_INT8:
            if not builder.platform_has_fast_int8:
                raise ValueError('TensorRT reports that this GPU has no fast INT8 support.')
        case _:
            raise ValueError(f'Cannot build a TensorRT engine for {backend.value}.')
    if not builder.platform_has_fast_fp16:
        raise ValueError('TensorRT reports that this GPU has no fast FP16 support.')
    serialized = builder.build_serialized_network(network, configuration)
    if serialized is None:
        raise ValueError(f'TensorRT failed to build the {backend.value} engine.')
    write_bytes_atomically(output_path, bytes(serialized))
    return (
        ArtifactIdentity(path=str(output_path), sha256=file_sha256(output_path)),
        builder.platform_has_fast_fp16,
        builder.platform_has_fast_int8,
    )


def _measure_runner(
    runner: _CudaGraphRunner,
    warmup_iterations: int,
    repetitions: int,
    iterations_per_repetition: int,
    device: torch.device,
) -> TimingDistribution:
    for _ in range(warmup_iterations):
        runner.replay()
    torch.cuda.synchronize(device)
    durations: list[float] = []
    for _ in range(repetitions):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations_per_repetition):
            runner.replay()
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
    return summarize_timings(tuple(durations), iterations_per_repetition, BATCH_SIZE)


def _candidate_measurement(
    backend: Backend,
    engine: ArtifactIdentity,
    runner: _CudaGraphRunner,
    reference_outputs: ModelOutputs,
    legal_action_mask: Tensor,
    arguments: BenchmarkArguments,
    device: torch.device,
) -> CandidateMeasurement:
    padded_outputs = runner.outputs()
    positions = reference_outputs.policy_logits.shape[0]
    outputs = ModelOutputs(
        policy_logits=padded_outputs.policy_logits[:positions],
        wdl_probabilities=padded_outputs.wdl_probabilities[:positions],
    )
    fidelity = measure_fidelity(reference_outputs, outputs, legal_action_mask)
    return CandidateMeasurement(
        backend=backend,
        engine=engine,
        timing=_measure_runner(
            runner,
            arguments.warmup_iterations,
            arguments.repetitions,
            arguments.iterations_per_repetition,
            device,
        ),
        fidelity=fidelity,
        fidelity_failures=fidelity_failures(fidelity, arguments.fidelity_limits),
    )


def _driver_version() -> str:
    completed = subprocess.run(
        ('nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'),
        check=True,
        capture_output=True,
        text=True,
    )
    versions = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if not versions:
        raise ValueError('nvidia-smi did not report a driver version.')
    return versions[0]


def _input_shape() -> tuple[int, int, int, int]:
    return (
        BATCH_SIZE,
        CHESS_NETWORK_DIMENSIONS.channels,
        CHESS_NETWORK_DIMENSIONS.rows,
        CHESS_NETWORK_DIMENSIONS.columns,
    )


def run_benchmark(arguments: BenchmarkArguments) -> TensorRtInferenceBenchmarkReport:
    if not arguments.acknowledge_gpu_load:
        raise ValueError('TensorRT engine building and benchmarking require --acknowledge-gpu-load.')
    if not torch.cuda.is_available():
        raise ValueError('The TensorRT benchmark requires CUDA.')
    if arguments.gpu_id < 0 or arguments.gpu_id >= torch.cuda.device_count():
        raise ValueError(f'GPU ID {arguments.gpu_id} is not available.')
    if arguments.warmup_iterations < 1 or arguments.repetitions < 3 or arguments.iterations_per_repetition < 1:
        raise ValueError('Use at least one warm-up, three repetitions, and one timed iteration per repetition.')
    if arguments.calibration_position_count < BATCH_SIZE or arguments.calibration_position_count % BATCH_SIZE != 0:
        raise ValueError(f'INT8 calibration requires a positive whole number of {BATCH_SIZE}-position batches.')
    if arguments.fidelity_position_offset < 0 or not 1 <= arguments.fidelity_position_count <= BATCH_SIZE:
        raise ValueError(f'Fidelity offset must be nonnegative and position count must lie in [1, {BATCH_SIZE}].')
    if not arguments.quantized_node_patterns:
        raise ValueError('At least one quantized node pattern is required.')

    device = torch.device('cuda', arguments.gpu_id)
    torch.cuda.set_device(device)
    checkpoint = _resolve_checkpoint(arguments)
    configuration = load_chess_experiment_configuration(arguments.configuration_path)
    loaded_states, loaded_legal_action_mask = load_positions(
        arguments.benchmark_dataset_path,
        arguments.fidelity_position_offset + arguments.fidelity_position_count,
    )
    benchmark_states = loaded_states[
        arguments.fidelity_position_offset : arguments.fidelity_position_offset + arguments.fidelity_position_count
    ].to(torch.int8)
    legal_action_mask = loaded_legal_action_mask[
        arguments.fidelity_position_offset : arguments.fidelity_position_offset + arguments.fidelity_position_count
    ]
    benchmark_dataset = _load_dataset_identity(arguments.benchmark_dataset_path, benchmark_states, legal_action_mask)
    runner_states = benchmark_states
    if benchmark_states.shape[0] < BATCH_SIZE:
        padding_indices = torch.arange(BATCH_SIZE - benchmark_states.shape[0]) % benchmark_states.shape[0]
        runner_states = torch.cat((benchmark_states, benchmark_states[padding_indices]), dim=0)
    calibration_states, calibration_source_identity = _load_calibration(
        arguments.calibration_source,
        arguments.calibration_position_count,
        configuration,
        runner_states,
    )
    overlap_count = _input_overlap_count(calibration_states, benchmark_states)
    if overlap_count:
        raise ValueError(
            f'Calibration contains {overlap_count} inputs from the 320-position fidelity workload; '
            'select a calibration source that is disjoint from the benchmark dataset.'
        )

    arguments.artifact_directory.mkdir(parents=True, exist_ok=True)
    float16_onnx_path = arguments.artifact_directory / f'{checkpoint.model_id}-batch320-fp16.onnx'
    int8_source_onnx_path = arguments.artifact_directory / f'{checkpoint.model_id}-batch320-fp32-for-int8.onnx'
    int8_qdq_onnx_path = arguments.artifact_directory / f'{checkpoint.model_id}-batch320-int8-qdq.onnx'
    float16_engine_path = arguments.artifact_directory / f'{checkpoint.model_id}-batch320-fp16.engine'
    int8_engine_path = arguments.artifact_directory / f'{checkpoint.model_id}-batch320-int8.engine'
    float16_onnx_artifact = _export_onnx(checkpoint.model_path, float16_onnx_path, torch.float16)
    int8_source_onnx_artifact = _export_onnx(checkpoint.model_path, int8_source_onnx_path, torch.float32)
    int8_qdq_onnx_artifact, quantize_linear_count, dequantize_linear_count, quantization_output = _quantize_onnx(
        int8_source_onnx_path,
        int8_qdq_onnx_path,
        calibration_states,
        device,
        arguments.calibration_method,
        arguments.quantized_node_patterns,
        arguments.autotune,
    )
    float16_engine, fast_float16, fast_int8 = _build_engine(
        float16_onnx_path,
        float16_engine_path,
        Backend.TENSORRT_FLOAT16,
    )
    int8_engine, _, _ = _build_engine(
        int8_qdq_onnx_path,
        int8_engine_path,
        Backend.TENSORRT_INT8,
    )

    reference_runner = _TorchScriptCudaGraphRunner(
        checkpoint.model_path, runner_states, device, arguments.warmup_iterations
    )
    padded_reference_outputs = reference_runner.outputs()
    reference_outputs = ModelOutputs(
        policy_logits=padded_reference_outputs.policy_logits[: arguments.fidelity_position_count],
        wdl_probabilities=padded_reference_outputs.wdl_probabilities[: arguments.fidelity_position_count],
    )
    reference = ReferenceMeasurement(
        timing=_measure_runner(
            reference_runner,
            arguments.warmup_iterations,
            arguments.repetitions,
            arguments.iterations_per_repetition,
            device,
        )
    )
    candidates = tuple(
        _candidate_measurement(
            backend,
            engine,
            _TensorRtCudaGraphRunner(engine_path, runner_states, device, arguments.warmup_iterations),
            reference_outputs,
            legal_action_mask,
            arguments,
            device,
        )
        for backend, engine, engine_path in (
            (Backend.TENSORRT_FLOAT16, float16_engine, float16_engine_path),
            (Backend.TENSORRT_INT8, int8_engine, int8_engine_path),
        )
    )
    properties = torch.cuda.get_device_properties(device)
    return TensorRtInferenceBenchmarkReport(
        source_revision=read_source_revision(),
        experiment_configuration_path=str(arguments.configuration_path),
        experiment_configuration_sha256=experiment_configuration_sha256(configuration),
        checkpoint_generation=arguments.checkpoint_generation,
        model_id=checkpoint.model_id,
        checkpoint_manifest=ArtifactIdentity(
            path=str(arguments.checkpoint_manifest_path), sha256=checkpoint.manifest_sha256
        ),
        inference_model=ArtifactIdentity(path=str(checkpoint.model_path), sha256=checkpoint.model_sha256),
        benchmark_dataset=benchmark_dataset,
        fidelity_position_offset=arguments.fidelity_position_offset,
        fidelity_position_count=arguments.fidelity_position_count,
        calibration=CalibrationIdentity(
            source=calibration_source_identity,
            requested_positions=arguments.calibration_position_count,
            submitted_batches=arguments.calibration_position_count // BATCH_SIZE,
            submitted_positions=arguments.calibration_position_count,
            algorithm=f'modelopt_onnx_ptq_{arguments.calibration_method.value}',
            tool_version=version('nvidia-modelopt'),
            quantized_node_patterns=arguments.quantized_node_patterns,
            autotune=arguments.autotune,
        ),
        input_shape=_input_shape(),
        warmup_iterations=arguments.warmup_iterations,
        hardware=HardwareDescription(
            gpu_id=arguments.gpu_id,
            device_name=properties.name,
            compute_capability=(properties.major, properties.minor),
            driver_version=_driver_version(),
            torch_version=torch.__version__,
            torch_cuda_version=torch.version.cuda or 'none',
            onnx_version=onnx.__version__,
            tensorrt_version=trt.__version__,
            tensorrt_fast_float16=fast_float16,
            tensorrt_bfloat16_builder_flag=trt.BuilderFlag.BF16 is not None,
            tensorrt_fast_int8=fast_int8,
            floating_precision_reason=(
                'FP16 is the conversion baseline because TensorRT 10.14 supports fast FP16 on SM 8.9 and '
                'provides the fallback precision around explicitly quantized INT8 convolution layers.'
            ),
        ),
        float16_onnx_model=float16_onnx_artifact,
        int8_source_onnx_model=int8_source_onnx_artifact,
        int8_qdq_onnx_model=int8_qdq_onnx_artifact,
        quantization_output=quantization_output,
        int8_quantize_linear_node_count=quantize_linear_count,
        int8_dequantize_linear_node_count=dequantize_linear_count,
        onnx_opset_version=ONNX_OPSET_VERSION,
        tensorrt_workspace_bytes=TENSORRT_WORKSPACE_BYTES,
        tensorrt_builder_optimization_level=TENSORRT_BUILDER_OPTIMIZATION_LEVEL,
        fidelity_limits=arguments.fidelity_limits,
        reference=reference,
        candidates=candidates,
    )


def parse_arguments() -> BenchmarkArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--checkpoint-manifest', type=Path, required=True)
    parser.add_argument('--checkpoint-generation', type=int, required=True)
    parser.add_argument('--benchmark-dataset', type=Path, required=True)
    calibration_group = parser.add_mutually_exclusive_group(required=True)
    calibration_group.add_argument('--calibration-dataset', type=Path)
    calibration_group.add_argument('--calibration-replay', type=Path)
    parser.add_argument('--calibration-random-seed', type=int, default=DEFAULT_CALIBRATION_RANDOM_SEED)
    parser.add_argument(
        '--calibration-method', type=CalibrationMethod, choices=tuple(CalibrationMethod), default=CalibrationMethod.MAX
    )
    parser.add_argument('--quantized-node-pattern', action='append', dest='quantized_node_patterns')
    parser.add_argument('--autotune', action='store_true')
    parser.add_argument('--fidelity-position-offset', type=int, default=0)
    parser.add_argument('--fidelity-position-count', type=int, default=BATCH_SIZE)
    parser.add_argument('--artifact-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--warmup-iterations', type=int, default=50)
    parser.add_argument('--repetitions', type=int, default=15)
    parser.add_argument('--iterations-per-repetition', type=int, default=100)
    parser.add_argument('--calibration-position-count', type=int, default=DEFAULT_CALIBRATION_POSITION_COUNT)
    parser.add_argument('--minimum-policy-top1-agreement', type=float, default=0.98)
    parser.add_argument('--maximum-mean-policy-kl-divergence', type=float, default=0.005)
    parser.add_argument('--maximum-wdl-mean-absolute-error', type=float, default=0.01)
    parser.add_argument('--maximum-expected-value-mean-absolute-error', type=float, default=0.015)
    parser.add_argument('--acknowledge-gpu-load', action='store_true')
    parsed = parser.parse_args()
    if parsed.calibration_dataset is not None:
        calibration_source: CalibrationSource = EvaluationDatasetCalibrationSource(parsed.calibration_dataset.resolve())
    else:
        calibration_source = ReplayCalibrationSource(
            parsed.calibration_replay.resolve(),
            parsed.calibration_random_seed,
        )
    return BenchmarkArguments(
        configuration_path=parsed.configuration.resolve(),
        checkpoint_manifest_path=parsed.checkpoint_manifest.resolve(),
        checkpoint_generation=parsed.checkpoint_generation,
        benchmark_dataset_path=parsed.benchmark_dataset.resolve(),
        calibration_source=calibration_source,
        artifact_directory=parsed.artifact_directory.resolve(),
        output_path=parsed.output.resolve(),
        gpu_id=parsed.gpu_id,
        warmup_iterations=parsed.warmup_iterations,
        repetitions=parsed.repetitions,
        iterations_per_repetition=parsed.iterations_per_repetition,
        calibration_position_count=parsed.calibration_position_count,
        calibration_method=parsed.calibration_method,
        quantized_node_patterns=tuple(parsed.quantized_node_patterns or INT8_QUANTIZED_NODE_PATTERNS),
        autotune=parsed.autotune,
        fidelity_position_offset=parsed.fidelity_position_offset,
        fidelity_position_count=parsed.fidelity_position_count,
        fidelity_limits=FidelityLimits(
            minimum_policy_top1_agreement=parsed.minimum_policy_top1_agreement,
            maximum_mean_policy_kl_divergence=parsed.maximum_mean_policy_kl_divergence,
            maximum_wdl_mean_absolute_error=parsed.maximum_wdl_mean_absolute_error,
            maximum_expected_value_mean_absolute_error=parsed.maximum_expected_value_mean_absolute_error,
        ),
        acknowledge_gpu_load=parsed.acknowledge_gpu_load,
    )


def main() -> None:
    arguments = parse_arguments()
    report = run_benchmark(arguments)
    write_text_atomically(arguments.output_path, report.model_dump_json(indent=2) + '\n')
    print(json.dumps(report.model_dump(mode='json'), indent=2))
    for candidate in report.candidates:
        validate_fidelity(candidate.backend.value, candidate.fidelity, report.fidelity_limits)


if __name__ == '__main__':
    main()
