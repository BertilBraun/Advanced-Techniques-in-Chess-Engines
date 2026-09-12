from __future__ import annotations

import json
from pathlib import Path

import modelopt.onnx.quantization as moq
import onnx
import tensorrt as trt
import torch

from tools.benchmark_tensorrt_inference import _TensorRtCudaGraphRunner, _measure_runner
from tools.measure_inference_precision_agreement import MemoryFormat, Precision, load_positions, run_variant
from tools.tensorrt_benchmark_metrics import measure_fidelity

BASE = Path('/workspace/int8-salvage')
SOURCE = BASE / 'calibration-sweep/max-4/artifacts/chess-cnn-14x160-fromto-batch320-fp32-for-int8.onnx'
MODEL = Path(
    '/workspace/alphazero-engine-v34-lr-001/py/training_data/production/'
    'vast-chess-8gpu-integrated-v34/model_1785.jit.pt'
)
DATASET = Path('/workspace/evaluation-artifacts/chess/chess-stockfish-evaluation-v33.bin')
ROOT = BASE / 'fp8-probe'


def build_engine(onnx_path: Path, engine_path: Path) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) | (
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise ValueError('\n'.join(str(parser.get_error(index)) for index in range(parser.num_errors)))
    configuration = builder.create_builder_config()
    configuration.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024**3)
    configuration.builder_optimization_level = 3
    serialized = builder.build_serialized_network(network, configuration)
    if serialized is None:
        raise ValueError('FP8 engine build failed.')
    engine_path.write_bytes(bytes(serialized))


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    device = torch.device('cuda', 0)
    states, legal_mask = load_positions(DATASET, 320)
    output = ROOT / 'model-fp8.onnx'
    moq.quantize(
        onnx_path=str(SOURCE),
        quantize_mode='fp8',
        calibration_data=states.float().numpy(),
        calibration_method='max',
        calibration_eps=['cuda:0', 'cpu'],
        op_types_to_quantize=['Conv'],
        nodes_to_quantize=['/(start_block|backbone).*Conv.*$'],
        nodes_to_exclude=['.*policy_head.*', '.*value_head.*'],
        high_precision_dtype='fp16',
        output_path=str(output),
    )
    onnx.checker.check_model(onnx.load(output), full_check=True)
    engine = ROOT / 'model-fp8.engine'
    build_engine(output, engine)
    runner = _TensorRtCudaGraphRunner(engine, states.to(torch.int8), device, 10)
    reference = run_variant(MODEL, states, Precision.BFLOAT16, MemoryFormat.CHANNELS_LAST, device, 320, True)
    report = {
        'fidelity': measure_fidelity(reference, runner.outputs(), legal_mask).model_dump(),
        'timing': _measure_runner(runner, 10, 3, 20, device).model_dump(),
    }
    (ROOT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
