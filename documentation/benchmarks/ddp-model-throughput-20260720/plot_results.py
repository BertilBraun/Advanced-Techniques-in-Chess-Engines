from __future__ import annotations

from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict


ARTIFACT_DIRECTORY = Path(__file__).resolve().parent
RESULTS_PATH = ARTIFACT_DIRECTORY / 'results.json'
PRODUCTION_RESULTS_PATH = ARTIFACT_DIRECTORY.parent / 'ddp-production-training-20260720' / 'results.json'


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


class ProductionHardware(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    gpu_count: int
    gpu_model: str
    gpu_memory_mib: int
    logical_cpu_count: int
    torch_version: str
    cuda_version: str


class ProductionGpuUtilization(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    rank: int
    device_id: int
    samples: int
    mean_sm_percent: float
    mean_memory_controller_percent: float
    peak_memory_mib: int


class IsolatedProductionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    source_revision: str
    method: str
    rank_device_ids: tuple[int, ...]
    global_batch_size: int
    local_batch_size: int
    replay_samples: int
    retained_samples: int
    dropped_incomplete_global_batch_samples: int
    optimizer_steps_per_rank: int
    samples_per_rank: int
    partition_overlap_samples: int
    duplicated_padding_samples: int
    aggregated_training_samples: int
    training_phase_seconds: float
    optimizer_seconds: float
    production_phase_samples_per_second: float
    optimizer_samples_per_second: float
    peak_aggregate_trainer_process_tree_rss_mib: float
    gpu_utilization: tuple[ProductionGpuUtilization, ...]


class SelfPlayContentionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    processes_per_gpu: int
    total_processes: int
    mcts_threads_per_process: int
    parallel_games_per_process: int
    full_searches: int
    fast_searches: int
    inference_cache_enabled: bool
    measurement_seconds: float
    completed_games: int
    completed_games_per_second: float
    generated_samples: int
    generated_samples_per_second: float
    retained_samples: int
    retained_samples_per_second: float
    completed_game_plies_per_second: float
    inference_evaluations: int
    model_calls: int
    average_inference_batch_size: float
    summed_worker_peak_rss_mib: float


class CombinedGpuUtilization(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    device_id: int
    mean_sm_percent: float
    peak_memory_mib: int


class ContendedProductionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    ddp_source_revision: str
    ddp_replay: str
    ddp_rank_device_ids: tuple[int, ...]
    ddp_global_batch_size: int
    ddp_local_batch_size: int
    ddp_optimizer_steps_per_rank: int
    ddp_samples_per_rank: int
    ddp_aggregated_training_samples: int
    ddp_training_phase_seconds: float
    ddp_optimizer_seconds: float
    ddp_production_phase_samples_per_second: float
    ddp_optimizer_samples_per_second: float
    ddp_peak_process_tree_rss_mib: float
    self_play: SelfPlayContentionResult
    combined_gpu_utilization: tuple[CombinedGpuUtilization, ...]
    peak_host_ram_percent: float


class PriorIsolatedReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    method: str
    four_gpu_ddp_samples_per_second: float
    four_gpu_data_parallel_samples_per_second: float


class ProductionBenchmarkResults(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    hardware: ProductionHardware
    isolated_real_replay: IsolatedProductionResult
    ddp_with_half_self_play: ContendedProductionResult
    prior_isolated_reference: PriorIsolatedReference


class ThroughputBar(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    label: str
    samples_per_second: float
    color: str


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


def select_result(
    results: BenchmarkResults,
    strategy: Strategy,
    device_count: int,
    global_batch_size: int,
) -> ThroughputResult:
    return next(
        result
        for result in results.results
        if result.strategy is strategy
        and len(result.devices) == device_count
        and result.global_batch_size == global_batch_size
    )


def plot_throughput(
    results: BenchmarkResults,
    production: ProductionBenchmarkResults,
) -> None:
    single_gpu = select_result(results, Strategy.SINGLE_GPU, 1, 1_024)
    data_parallel = select_result(results, Strategy.DATA_PARALLEL, 4, 2_048)
    distributed_data_parallel = select_result(
        results,
        Strategy.DISTRIBUTED_DATA_PARALLEL,
        4,
        2_048,
    )
    bars_to_plot = tuple(
        reversed(
            (
                ThroughputBar(
                    label='Single GPU, synthetic model path',
                    samples_per_second=single_gpu.samples_per_second,
                    color=STRATEGY_COLORS[Strategy.SINGLE_GPU],
                ),
                ThroughputBar(
                    label='DataParallel, synthetic model path',
                    samples_per_second=data_parallel.samples_per_second,
                    color=STRATEGY_COLORS[Strategy.DATA_PARALLEL],
                ),
                ThroughputBar(
                    label='DDP, short synthetic model path',
                    samples_per_second=distributed_data_parallel.samples_per_second,
                    color='#8BCB82',
                ),
                ThroughputBar(
                    label='DDP, sustained synthetic model path',
                    samples_per_second=results.steady_state_four_gpu_ddp.samples_per_second,
                    color='#68B35F',
                ),
                ThroughputBar(
                    label='DDP, production real replay',
                    samples_per_second=production.isolated_real_replay.production_phase_samples_per_second,
                    color='#3B8F45',
                ),
                ThroughputBar(
                    label='DDP + half self-play',
                    samples_per_second=production.ddp_with_half_self_play.ddp_production_phase_samples_per_second,
                    color='#1F6F78',
                ),
            )
        )
    )
    labels = tuple(result.label for result in bars_to_plot)
    throughput = tuple(result.samples_per_second / 1_000 for result in bars_to_plot)
    colors = tuple(result.color for result in bars_to_plot)

    figure, axis = plt.subplots(figsize=(9.5, 5.2))
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
    axis.set_xlabel('Training throughput (thousand samples/s)')
    axis.set_title('Production data loading and concurrent self-play reduce DDP throughput')
    axis.grid(axis='x')
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc='lower right')
    figure.text(
        0.01,
        -0.02,
        (
            '12×112 network, BF16. Synthetic ranks used duplicate resident samples; '
            'production ranks used disjoint replay partitions.'
        ),
        color='#555555',
        fontsize=8.5,
    )
    save_figure(figure, 'training-throughput')


def plot_utilization(production: ProductionBenchmarkResults) -> None:
    isolated_by_device = tuple(
        sorted(
            production.isolated_real_replay.gpu_utilization,
            key=lambda sample: sample.device_id,
        )
    )
    contended_by_device = tuple(
        sorted(
            production.ddp_with_half_self_play.combined_gpu_utilization,
            key=lambda sample: sample.device_id,
        )
    )
    devices = tuple(f'GPU {sample.device_id}' for sample in isolated_by_device)
    isolated_means = tuple(sample.mean_sm_percent for sample in isolated_by_device)
    contended_means = tuple(sample.mean_sm_percent for sample in contended_by_device)
    positions = tuple(float(index) for index in range(len(devices)))
    width = 0.36

    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    isolated_bars = axis.bar(
        tuple(position - width / 2 for position in positions),
        isolated_means,
        color='#54A24B',
        width=width,
        label='Production DDP',
    )
    contended_bars = axis.bar(
        tuple(position + width / 2 for position in positions),
        contended_means,
        color='#1F6F78',
        width=width,
        label='DDP + half self-play',
    )
    axis.bar_label(isolated_bars, labels=[f'{value:.1f}%' for value in isolated_means], padding=4)
    axis.bar_label(contended_bars, labels=[f'{value:.1f}%' for value in contended_means], padding=4)
    axis.set_xticks(positions, devices)
    axis.set_ylim(0, 125)
    axis.set_ylabel('Mean SM utilization (%)')
    axis.set_title('Half self-play fills the compute headroom during DDP training')
    axis.grid(axis='y')
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc='upper center', ncol=2)
    figure.text(
        0.01,
        -0.02,
        (
            'Isolated DDP used real replay data (127 samples/device); the contention run used '
            '5 self-play processes/GPU and reached 96.6–97.1% mean SM utilization.'
        ),
        color='#555555',
        fontsize=8.5,
    )
    save_figure(figure, 'ddp-gpu-utilization')


def main() -> None:
    configure_plot_style()
    results = BenchmarkResults.model_validate_json(RESULTS_PATH.read_text(encoding='utf-8'))
    production = ProductionBenchmarkResults.model_validate_json(PRODUCTION_RESULTS_PATH.read_text(encoding='utf-8'))
    plot_throughput(results, production)
    plot_utilization(production)


if __name__ == '__main__':
    main()
