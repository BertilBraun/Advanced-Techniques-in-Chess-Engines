from __future__ import annotations

import json
from pathlib import Path

import onnxruntime as ort
import torch

from tools.benchmark_tensorrt_inference import BATCH_SIZE, _TensorRtCudaGraphRunner
from tools.measure_inference_precision_agreement import load_positions
from tools.tensorrt_benchmark_metrics import ModelOutputs, measure_fidelity

BASE = Path('/workspace/int8-salvage')
DATASET = Path('/workspace/evaluation-artifacts/chess/chess-stockfish-evaluation-v33.bin')
OUTPUT = BASE / 'progressive-runtime-fidelity.json'

ROOTS = {
    'single_early_conv': BASE / 'layer-sweep/01/artifacts',
    'ranked_4': BASE / 'partition-sweep/ranked-04/artifacts',
    'ranked_8': BASE / 'partition-sweep/ranked-08/artifacts',
    'contiguous_blocks_0_3': BASE / 'partition-sweep/contiguous-blocks-0-3/artifacts',
    'ranked_12': BASE / 'partition-sweep/ranked-12/artifacts',
    'ranked_20': BASE / 'partition-sweep/ranked-20/artifacts',
    'ranked_28': BASE / 'partition-sweep/ranked-28/artifacts',
    'full_trunk': BASE / 'calibration-sweep/max-4/artifacts',
}


def find(root: Path, suffix: str) -> Path:
    matches = tuple(root.glob(suffix))
    if len(matches) != 1:
        raise ValueError(f'Expected one {suffix} under {root}, found {matches}.')
    return matches[0]


def ort_outputs(path: Path, states: torch.Tensor) -> tuple[ModelOutputs, tuple[str, ...]]:
    session = ort.InferenceSession(
        str(path), providers=[('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    )
    policy, wdl = session.run(None, {'states': states.to(torch.float32).numpy()})
    return ModelOutputs(torch.from_numpy(policy).float(), torch.from_numpy(wdl).float()), tuple(session.get_providers())


def fidelity(
    reference: ModelOutputs, candidate: ModelOutputs, legal_mask: torch.Tensor, count: int
) -> dict[str, float | int]:
    selected_reference = ModelOutputs(reference.policy_logits[:count], reference.wdl_probabilities[:count])
    selected_candidate = ModelOutputs(candidate.policy_logits[:count], candidate.wdl_probabilities[:count])
    return measure_fidelity(selected_reference, selected_candidate, legal_mask[:count]).model_dump()


def main() -> None:
    count = 196
    states, legal_mask = load_positions(DATASET, BATCH_SIZE)
    device = torch.device('cuda', 0)
    rows: dict[str, object] = {}
    for name, root in ROOTS.items():
        qdq_path = find(root, '*-int8-qdq.onnx')
        source_path = find(root, '*-fp32-for-int8.onnx')
        engine_path = find(root, '*-int8.engine')
        source, source_providers = ort_outputs(source_path, states)
        qdq, qdq_providers = ort_outputs(qdq_path, states)
        tensorrt = _TensorRtCudaGraphRunner(engine_path, states.to(torch.int8), device, 10).outputs()
        rows[name] = {
            'qdq_node_count': sum(node.op_type == 'QuantizeLinear' for node in __import__('onnx').load(qdq_path).graph.node),
            'source_ort_providers': source_providers,
            'qdq_ort_providers': qdq_providers,
            'source_ort_vs_qdq_ort': fidelity(source, qdq, legal_mask, count),
            'qdq_ort_vs_tensorrt': fidelity(qdq, tensorrt, legal_mask, count),
            'source_ort_vs_tensorrt': fidelity(source, tensorrt, legal_mask, count),
        }
        print(name, json.dumps(rows[name]), flush=True)
    OUTPUT.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
