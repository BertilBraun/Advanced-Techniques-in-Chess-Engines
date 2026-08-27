from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.util.log import log

WARMUP_FORWARDS = 20
MEASURED_FORWARDS = 200


@dataclass(frozen=True)
class ForwardThroughput:
    cell: str
    batch_size: int
    repeats: int
    mean_positions_per_second: float
    minimum_positions_per_second: float
    maximum_positions_per_second: float


def measure(model: torch.jit.ScriptModule, batch_size: int, device: torch.device, repeats: int) -> list[float]:
    inputs = torch.zeros(
        (
            batch_size,
            CHESS_NETWORK_DIMENSIONS.channels,
            CHESS_NETWORK_DIMENSIONS.rows,
            CHESS_NETWORK_DIMENSIONS.columns,
        ),
        device=device,
        dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
    )
    rates: list[float] = []
    with torch.no_grad():
        for _ in range(WARMUP_FORWARDS):
            model(inputs)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        for _ in range(repeats):
            started_at = time.perf_counter()
            for _ in range(MEASURED_FORWARDS):
                model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            rates.append(MEASURED_FORWARDS * batch_size / (time.perf_counter() - started_at))
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Forward-pass throughput of exported models, with no search and so no dependence on tree shape.'
    )
    parser.add_argument('--run-state-root', required=True, type=Path)
    parser.add_argument('--cells', required=True, nargs='+')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch-sizes', default=(64,), nargs='+', type=int)
    parser.add_argument('--generation', default=322, type=int)
    parser.add_argument('--repeats', default=5, type=int)
    parser.add_argument('--device-id', default=0, type=int)
    namespace = parser.parse_args()

    device = torch.device('cuda', namespace.device_id) if torch.cuda.is_available() else torch.device('cpu')
    results: list[ForwardThroughput] = []
    for batch_size in namespace.batch_sizes:
        for cell in namespace.cells:
            path = namespace.run_state_root / cell / f'model_{namespace.generation}.jit.pt'
            model = torch.jit.load(str(path), map_location=device)
            # The native pipeline converts only the floating-point state, so the benchmark does the same.
            for tensors in (model.named_parameters(), model.named_buffers()):
                for _, tensor in tensors:
                    if tensor.is_floating_point() and device.type == 'cuda':
                        tensor.data = tensor.data.to(torch.bfloat16)
            model.eval()
            model = torch.jit.freeze(model)
            rates = measure(model, batch_size, device, namespace.repeats)
            results.append(
                ForwardThroughput(
                    cell=cell,
                    batch_size=batch_size,
                    repeats=namespace.repeats,
                    mean_positions_per_second=statistics.mean(rates),
                    minimum_positions_per_second=min(rates),
                    maximum_positions_per_second=max(rates),
                )
            )
            log(
                f'batch {batch_size:5d} {cell:26s} {statistics.mean(rates):10.0f} positions/s '
                f'(min {min(rates):.0f}, max {max(rates):.0f})'
            )
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    namespace.output.parent.mkdir(parents=True, exist_ok=True)
    namespace.output.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding='utf-8')
    log(f'Wrote {namespace.output}.')


if __name__ == '__main__':
    main()
