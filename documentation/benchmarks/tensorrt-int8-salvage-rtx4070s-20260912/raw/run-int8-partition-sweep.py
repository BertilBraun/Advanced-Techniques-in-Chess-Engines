from __future__ import annotations

import concurrent.futures
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path('/workspace/int8-salvage/partition-sweep')
LAYER_ROOT = Path('/workspace/int8-salvage/layer-sweep')
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


def layer_ranking() -> tuple[str, ...]:
    rows: list[tuple[float, float, str]] = []
    for report_path in LAYER_ROOT.glob('*/report.json'):
        report = json.loads(report_path.read_text())
        candidate = next(candidate for candidate in report['candidates'] if candidate['backend'] == 'tensorrt_int8_calibrated')
        node_name = (report_path.parent / 'node.txt').read_text().strip()
        if 'start_block' in node_name:
            continue
        fidelity = candidate['fidelity']
        rows.append((-fidelity['policy_top1_agreement'], fidelity['mean_policy_kl_divergence'], node_name))
    return tuple(row[2] for row in sorted(rows))


def run_partition(index_and_partition: tuple[int, tuple[str, tuple[str, ...]]]) -> tuple[str, int]:
    index, (label, nodes) = index_and_partition
    output = ROOT / label
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = str(index % 8)
    patterns = tuple(argument for node in nodes for argument in ('--quantized-node-pattern', re.escape(node) + '$'))
    command = (
        *COMMON,
        *patterns,
        '--artifact-directory',
        str(output / 'artifacts'),
        '--output',
        str(output / 'report.json'),
    )
    (output / 'nodes.json').write_text(json.dumps(nodes, indent=2) + '\n')
    with (output / 'stdout.log').open('w') as stdout, (output / 'stderr.log').open('w') as stderr:
        completed = subprocess.run(command, cwd='/workspace/alphazero-engine/py', env=environment, stdout=stdout, stderr=stderr)
    return label, completed.returncode


def main() -> None:
    ranking = layer_ranking()
    contiguous_early = tuple(
        node for node in ranking if any(f'/backbone.{index}/' in node for index in range(4))
    )
    partitions = tuple((f'ranked-{count:02d}', ranking[:count]) for count in (4, 8, 12, 16, 20, 24, 28)) + (
        ('contiguous-blocks-0-3', contiguous_early),
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for result in executor.map(run_partition, enumerate(partitions)):
            print(result, flush=True)


if __name__ == '__main__':
    main()
