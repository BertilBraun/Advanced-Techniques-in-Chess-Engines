from __future__ import annotations

import concurrent.futures
import os
import re
import subprocess
from pathlib import Path

import onnx

ROOT = Path('/workspace/int8-salvage/layer-sweep')
SOURCE = Path(
    '/workspace/tensorrt-v34-terminal-explicit-qdq-one-conv-v2/'
    'chess-cnn-14x160-fromto-batch320-fp32-for-int8.onnx'
)
PYTHON = '/workspace/alphazero-engine-venv/bin/python'
COMMON = (
    PYTHON,
    '-m',
    'tools.benchmark_tensorrt_inference',
    '--configuration',
    '/workspace/run-control/configs/vast-chess-8gpu-integrated-v34-resume-g1702.yaml',
    '--checkpoint-manifest',
    '/workspace/alphazero-engine-v34-lr-001/py/training_data/production/'
    'vast-chess-8gpu-integrated-v34/checkpoint_1785.json',
    '--checkpoint-generation',
    '1785',
    '--benchmark-dataset',
    '/workspace/evaluation-artifacts/chess/chess-stockfish-evaluation-v33.bin',
    '--calibration-replay',
    '/workspace/alphazero-engine-v34-lr-001/py/training_data/production/'
    'vast-chess-8gpu-integrated-v34/replay.bin',
    '--calibration-position-count',
    '32000',
    '--calibration-random-seed',
    '2026091204',
    '--calibration-method',
    'max',
    '--fidelity-position-offset',
    '0',
    '--fidelity-position-count',
    '196',
    '--gpu-id',
    '0',
    '--warmup-iterations',
    '10',
    '--repetitions',
    '3',
    '--iterations-per-repetition',
    '20',
    '--acknowledge-gpu-load',
)


def run_node(index_and_name: tuple[int, str]) -> tuple[str, int]:
    index, node_name = index_and_name
    output = ROOT / f'{index:02d}'
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = str(index % 8)
    command = (
        *COMMON,
        '--quantized-node-pattern',
        re.escape(node_name) + '$',
        '--artifact-directory',
        str(output / 'artifacts'),
        '--output',
        str(output / 'report.json'),
    )
    with (output / 'stdout.log').open('w') as stdout, (output / 'stderr.log').open('w') as stderr:
        completed = subprocess.run(command, cwd='/workspace/alphazero-engine/py', env=environment, stdout=stdout, stderr=stderr)
    (output / 'node.txt').write_text(node_name + '\n')
    return node_name, completed.returncode


def main() -> None:
    model = onnx.load(SOURCE)
    node_names = tuple(node.name for node in model.graph.node if node.op_type == 'Conv' and 'value_head' not in node.name)
    ROOT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for result in executor.map(run_node, enumerate(node_names)):
            print(result, flush=True)


if __name__ == '__main__':
    main()
