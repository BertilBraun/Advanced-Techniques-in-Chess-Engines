from __future__ import annotations

from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict


ARTIFACT_DIRECTORY = Path(__file__).resolve().parent
RESULTS_PATH = ARTIFACT_DIRECTORY / 'results.json'


class Strategy(str, Enum):
    SINGLE_GPU = 'single_gpu'
    DATA_PARALLEL = 'data_parallel'
    DISTRIBUTED_DATA_PARALLEL = 'distributed_data_parallel'


class NetworkDescription(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    layers: int
    hidden_size: int
    parameters: int
    precision: str


class HardwareDescription(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    gpu_count: int
    gpu_model: str


class GpuUtilization(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    device: int
    sm_mean_percent: float
    sm_min_percent: float
    sm_max_percent: float


class MemoryControllerUtilization(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    observed_minimum: float
    observed_maximum: float


class SteadyStateResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    batches: int
    warmup_batches: int
    seconds_per_batch: float
    samples_per_second: float
    utilization_sampling_interval_seconds: float
    active_utilization_samples_per_gpu: int
    gpu_utilization: tuple[GpuUtilization, ...]
    memory_controller_utilization_percent: MemoryControllerUtilization


class ThroughputResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    strategy: Strategy
    devices: tuple[int, ...]
    local_batch_size: int
    global_batch_size: int
    seconds_per_batch: float
    samples_per_second: float


class BenchmarkResults(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    source_revision: str
    network: NetworkDescription
    hardware: HardwareDescription
    method: str
    steady_state_four_gpu_ddp: SteadyStateResult
    results: tuple[ThroughputResult, ...]


STRATEGY_COLORS = {
    Strategy.SINGLE_GPU: '#4C78A8',
    Strategy.DATA_PARALLEL: '#F58518',
    Strategy.DISTRIBUTED_DATA_PARALLEL: '#54A24B',
}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.titleweight': 'bold',
            'font.size': 10,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'grid.color': '#D9D9D9',
            'grid.linewidth': 0.7,
        }
    )


def save_figure(figure: Figure, stem: str) -> None:
    figure.savefig(ARTIFACT_DIRECTORY / f'{stem}.png', dpi=200, bbox_inches='tight')
    figure.savefig(ARTIFACT_DIRECTORY / f'{stem}.svg', bbox_inches='tight')
    plt.close(figure)


def result_label(result: ThroughputResult) -> str:
    match result.strategy:
        case Strategy.SINGLE_GPU:
            strategy = 'Single GPU'
        case Strategy.DATA_PARALLEL:
            strategy = 'DataParallel'
        case Strategy.DISTRIBUTED_DATA_PARALLEL:
            strategy = 'DDP'
    return f'{strategy}, {len(result.devices)} GPU{"s" if len(result.devices) > 1 else ""}, batch {result.global_batch_size:,}'


def plot_throughput(results: BenchmarkResults) -> None:
    ordered_results = tuple(reversed(results.results))
    labels = tuple(result_label(result) for result in ordered_results)
    throughput = tuple(result.samples_per_second / 1_000 for result in ordered_results)
    colors = tuple(STRATEGY_COLORS[result.strategy] for result in ordered_results)

    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    bars = axis.barh(labels, throughput, color=colors, height=0.66)
    baseline = next(
        result.samples_per_second / 1_000
        for result in results.results
        if result.strategy is Strategy.SINGLE_GPU and result.global_batch_size == 1_024
    )
    axis.axvline(
        baseline,
        color='#4C78A8',
        linewidth=1.4,
        linestyle='--',
        alpha=0.8,
        label='Single-GPU batch-1,024 baseline',
    )
    axis.bar_label(bars, labels=[f'{value:.1f}k' for value in throughput], padding=4)
    axis.set_xlim(0, max(throughput) * 1.16)
    axis.set_xlabel('Training throughput (thousand synthetic samples/s)')
    axis.set_title('DDP scales; single-process DataParallel does not')
    axis.grid(axis='x')
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc='lower right')
    figure.text(
        0.01,
        -0.02,
        '12×112 network, BF16, resident synthetic batches; DDP ranks intentionally received duplicate samples.',
        color='#555555',
        fontsize=8.5,
    )
    save_figure(figure, 'training-throughput')


def plot_utilization(results: BenchmarkResults) -> None:
    steady_state = results.steady_state_four_gpu_ddp
    utilization = steady_state.gpu_utilization
    devices = tuple(f'GPU {sample.device}' for sample in utilization)
    means = tuple(sample.sm_mean_percent for sample in utilization)
    lower_errors = tuple(sample.sm_mean_percent - sample.sm_min_percent for sample in utilization)
    upper_errors = tuple(sample.sm_max_percent - sample.sm_mean_percent for sample in utilization)
    overall_mean = sum(means) / len(means)

    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    bars = axis.bar(
        devices,
        means,
        color='#54A24B',
        width=0.62,
        yerr=(lower_errors, upper_errors),
        capsize=5,
        error_kw={'elinewidth': 1.2, 'ecolor': '#285B2D'},
    )
    axis.bar_label(bars, labels=[f'{value:.1f}%' for value in means], padding=8)
    axis.axhline(
        overall_mean,
        color='#333333',
        linewidth=1.2,
        linestyle='--',
        label=f'Four-GPU mean: {overall_mean:.1f}%',
    )
    memory = steady_state.memory_controller_utilization_percent
    axis.axhspan(
        memory.observed_minimum,
        memory.observed_maximum,
        color='#B9D7EA',
        alpha=0.55,
        label=f'Memory-controller range: {memory.observed_minimum:.0f}–{memory.observed_maximum:.0f}%',
    )
    axis.set_ylim(0, 85)
    axis.set_ylabel('Utilization (%)')
    axis.set_title('Sustained four-GPU DDP leaves room for concurrent self-play')
    axis.grid(axis='y')
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))
    figure.text(
        0.01,
        -0.02,
        (
            f'{steady_state.batches} measured batches; '
            f'{steady_state.active_utilization_samples_per_gpu} one-second samples per GPU; '
            f'{steady_state.samples_per_second / 1_000:.1f}k synthetic samples/s sustained.'
        ),
        color='#555555',
        fontsize=8.5,
    )
    save_figure(figure, 'ddp-gpu-utilization')


def main() -> None:
    configure_plot_style()
    results = BenchmarkResults.model_validate_json(RESULTS_PATH.read_text(encoding='utf-8'))
    plot_throughput(results)
    plot_utilization(results)


if __name__ == '__main__':
    main()
