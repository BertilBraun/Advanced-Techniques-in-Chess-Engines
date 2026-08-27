from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.training.network import (
    DensePolicyHeadConfiguration,
    GlobalPoolingResidualContext,
    InferenceNetwork,
    Network,
    NetworkParams,
    ResidualContextPlacement,
)
from src.util.log import log

WARMUP_FORWARDS = 20
MEASURED_FORWARDS = 100


@dataclass(frozen=True)
class ChannelThroughput:
    hidden_size: int
    layers: int
    batch_size: int
    trunk_parameters: int
    positions_per_second: float
    multiply_accumulate_ratio_to_first: float
    throughput_ratio_to_first: float


def build(layers: int, hidden_size: int, device: torch.device, channels_last: bool = False) -> torch.jit.ScriptModule:
    architecture = NetworkParams(
        num_layers=layers,
        hidden_size=hidden_size,
        residual_context=GlobalPoolingResidualContext(placement=ResidualContextPlacement.EVERY_SECOND_BLOCK),
        policy_head=DensePolicyHeadConfiguration(channels=4),
        num_value_channels=2,
        value_fc_size=48,
    )
    model = Network(architecture, device, CHESS_NETWORK_DIMENSIONS)
    export = InferenceNetwork(model)
    export.eval()
    export.fuse_model()
    if channels_last:
        export = export.to(memory_format=torch.channels_last)
    scripted = torch.jit.script(export)
    if device.type == 'cuda':
        for tensors in (scripted.named_parameters(), scripted.named_buffers()):
            for _, tensor in tensors:
                if tensor.is_floating_point():
                    tensor.data = tensor.data.to(torch.bfloat16)
    return torch.jit.freeze(scripted.eval())


def trunk_parameter_count(layers: int, hidden_size: int) -> int:
    # Freezing inlines the parameters as constants, so the count has to come from an unfrozen model.
    architecture = NetworkParams(
        num_layers=layers,
        hidden_size=hidden_size,
        residual_context=GlobalPoolingResidualContext(placement=ResidualContextPlacement.EVERY_SECOND_BLOCK),
        policy_head=DensePolicyHeadConfiguration(channels=4),
        num_value_channels=2,
        value_fc_size=48,
    )
    model = Network(architecture, torch.device('cpu'), CHESS_NETWORK_DIMENSIONS)
    return sum(
        parameter.numel()
        for module in (model.start_block, model.backbone, model.finish_block)
        for parameter in module.parameters()
    )


def measure(
    model: torch.jit.ScriptModule,
    batch_size: int,
    device: torch.device,
    repeats: int,
    channels_last: bool = False,
) -> float:
    inputs = torch.zeros(
        (batch_size, CHESS_NETWORK_DIMENSIONS.channels, 8, 8),
        device=device,
        dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
    )
    if channels_last:
        inputs = inputs.to(memory_format=torch.channels_last)
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
    return statistics.median(rates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convolutional trunk throughput against channel count, to expose kernel alignment cliffs.'
    )
    parser.add_argument('--hidden-sizes', nargs='+', type=int, help='Widths to sweep at a single depth.')
    parser.add_argument('--layers', default=12, type=int)
    parser.add_argument(
        '--shapes',
        nargs='+',
        help='LAYERSxWIDTH shapes to measure in the given order, for comparing whole rungs.',
    )
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--repeats', default=5, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device-id', default=0, type=int)
    parser.add_argument('--cudnn-benchmark', action='store_true', help='Let cuDNN autotune each shape.')
    parser.add_argument(
        '--channels-last',
        action='store_true',
        help='Run the trunk in the NHWC layout the tensor cores actually want.',
    )
    namespace = parser.parse_args()

    torch.backends.cudnn.benchmark = namespace.cudnn_benchmark
    device = torch.device('cuda', namespace.device_id) if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(20260827)
    if not namespace.hidden_sizes and not namespace.shapes:
        raise ValueError('Give either --hidden-sizes or --shapes.')
    shapes = (
        [(int(shape.split('x')[0]), int(shape.split('x')[1])) for shape in namespace.shapes]
        if namespace.shapes
        else [(namespace.layers, hidden) for hidden in namespace.hidden_sizes]
    )

    results: list[ChannelThroughput] = []
    first_rate: float | None = None
    first_hidden: int | None = None
    for layers, hidden_size in shapes:
        model = build(layers, hidden_size, device, namespace.channels_last)
        rate = measure(model, namespace.batch_size, device, namespace.repeats, namespace.channels_last)
        first_rate = rate if first_rate is None else first_rate
        first_hidden = hidden_size if first_hidden is None else first_hidden
        trunk_parameters = trunk_parameter_count(layers, hidden_size)
        results.append(
            ChannelThroughput(
                hidden_size=hidden_size,
                layers=layers,
                batch_size=namespace.batch_size,
                trunk_parameters=trunk_parameters,
                positions_per_second=rate,
                # A trunk's cost scales with the square of its width, so this is what alignment-free
                # hardware would deliver relative to the first width measured.
                multiply_accumulate_ratio_to_first=(first_hidden / hidden_size) ** 2,
                throughput_ratio_to_first=rate / first_rate,
            )
        )
        log(
            f'{layers:3d}x{hidden_size:<4d} ({hidden_size % 32:2d} mod 32)  {rate:9.0f} positions/s  '
            f'{rate / first_rate:.3f} of the first shape  trunk {trunk_parameters:,}'
        )
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    namespace.output.parent.mkdir(parents=True, exist_ok=True)
    namespace.output.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding='utf-8')
    log(f'Wrote {namespace.output}.')


if __name__ == '__main__':
    main()
