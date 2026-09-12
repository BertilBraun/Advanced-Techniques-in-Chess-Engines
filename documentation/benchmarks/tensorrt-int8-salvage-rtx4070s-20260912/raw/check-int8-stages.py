from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from tools.benchmark_tensorrt_inference import _TensorRtCudaGraphRunner
from tools.measure_inference_precision_agreement import MemoryFormat, Precision, load_positions, run_variant
from tools.tensorrt_benchmark_metrics import ModelOutputs, measure_fidelity

ROOT = Path('/workspace/int8-salvage/calibration-sweep/max-4/artifacts')
SOURCE = ROOT / 'chess-cnn-14x160-fromto-batch320-fp32-for-int8.onnx'
QDQ = ROOT / 'chess-cnn-14x160-fromto-batch320-int8-qdq.onnx'
ENGINE = ROOT / 'chess-cnn-14x160-fromto-batch320-int8.engine'
MODEL = Path(
    '/workspace/alphazero-engine-v34-lr-001/py/training_data/production/'
    'vast-chess-8gpu-integrated-v34/model_1785.jit.pt'
)
DATASET = Path('/workspace/evaluation-artifacts/chess/chess-stockfish-evaluation-v33.bin')
OUTPUT = Path('/workspace/int8-salvage/stage-fidelity.json')


def ort_outputs(path: Path, states: torch.Tensor) -> ModelOutputs:
    session = ort.InferenceSession(
        str(path), providers=[('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    )
    policy, wdl = session.run(None, {'states': states.to(torch.float32).numpy()})
    return ModelOutputs(torch.from_numpy(policy).float(), torch.from_numpy(wdl).float())


def metrics(reference: ModelOutputs, candidate: ModelOutputs, legal_mask: torch.Tensor) -> dict[str, float | int]:
    return measure_fidelity(reference, candidate, legal_mask).model_dump()


def activation_errors(source_path: Path, qdq_path: Path, states: torch.Tensor) -> list[dict[str, float | str]]:
    source = onnx.shape_inference.infer_shapes(onnx.load(source_path))
    qdq = onnx.shape_inference.infer_shapes(onnx.load(qdq_path))
    qdq_values = {value.name: value for value in (*qdq.graph.value_info, *qdq.graph.output)}
    source_values = {value.name: value for value in (*source.graph.value_info, *source.graph.output)}
    names = [
        node.output[0]
        for node in source.graph.node
        if node.op_type == 'Conv' and 'value_head' not in node.name and node.output[0] in qdq_values
    ]
    for model, values in ((source, source_values), (qdq, qdq_values)):
        existing = {output.name for output in model.graph.output}
        model.graph.output.extend(values[name] for name in names if name not in existing)
    source_session = ort.InferenceSession(
        source.SerializeToString(), providers=[('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    )
    qdq_session = ort.InferenceSession(
        qdq.SerializeToString(), providers=[('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    )
    inputs = {'states': states.to(torch.float32).numpy()}
    source_outputs = source_session.run(names, inputs)
    qdq_outputs = qdq_session.run(names, inputs)
    rows: list[dict[str, float | str]] = []
    for name, reference, candidate in zip(names, source_outputs, qdq_outputs, strict=True):
        difference = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
        rows.append(
            {
                'tensor': name,
                'mean_absolute_error': float(difference.mean()),
                'maximum_absolute_error': float(difference.max()),
                'reference_mean_absolute_value': float(np.abs(reference.astype(np.float32)).mean()),
            }
        )
    return rows


def main() -> None:
    device = torch.device('cuda', 0)
    states, legal_mask = load_positions(DATASET, 320)
    reference = run_variant(MODEL, states, Precision.BFLOAT16, MemoryFormat.CHANNELS_LAST, device, 320, True)
    source = ort_outputs(SOURCE, states)
    qdq = ort_outputs(QDQ, states)
    tensorrt = _TensorRtCudaGraphRunner(ENGINE, states.to(torch.int8), device, 10).outputs()
    report = {
        'torchscript_bf16_vs_unquantized_onnx_fp32': metrics(reference, source, legal_mask),
        'unquantized_onnx_fp32_vs_qdq_onnxruntime': metrics(source, qdq, legal_mask),
        'qdq_onnxruntime_vs_tensorrt': metrics(qdq, tensorrt, legal_mask),
        'torchscript_bf16_vs_tensorrt': metrics(reference, tensorrt, legal_mask),
        'activation_errors': activation_errors(SOURCE, QDQ, states),
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
