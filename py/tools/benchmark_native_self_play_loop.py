from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import AlphaZeroCpp as native
import torch


@dataclass(frozen=True)
class LoopArguments:
    games: int
    parallel_searches: int
    baseline_visits: int
    search_budget_blend: float
    retained_root_visit_fraction: float
    inference_workers: int
    inference_batch_size: int
    outstanding_batches_per_worker: int
    warmup_batches: int
    measured_batches: int
    maximum_game_plies: int
    opening_plies: int
    greedy_after_ply: int
    inference_hidden: int
    collect_statistics: bool
    inference_device: str
    device_id: int
    sdpa_backend: str
    seed: int


class StubNetwork(torch.nn.Module):
    """Position-dependent but nearly free stand-in for the trained network."""

    def __init__(self, actions: int, channels: int, rows: int, columns: int, hidden: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(17)
        self.register_buffer('action_logits', torch.randn(actions, generator=generator) * 1.5)
        self.hidden = hidden
        inputs = channels * rows * columns
        self.first = torch.nn.Linear(inputs, max(1, hidden))
        self.second = torch.nn.Linear(max(1, hidden), actions)

    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = encoded.shape[0]
        activation = encoded.to(torch.float32)
        summary = activation.mean(dim=(1, 2, 3))
        if self.hidden > 0:
            logits = self.second(torch.relu(self.first(activation.reshape(batch, -1))))
        else:
            logits = self.action_logits.unsqueeze(0).expand(batch, -1) + summary.unsqueeze(1)
        win = torch.sigmoid(summary * 4.0) * 0.9
        loss = 0.9 - win
        draw = torch.full_like(win, 0.1)
        outcomes = torch.stack((win, draw, loss), dim=1)
        search_budget_logits = torch.zeros((batch, 1), dtype=torch.float32)
        return logits, outcomes, search_budget_logits


def write_stub_model(destination: Path, hidden: int) -> Path:
    dimensions = native.ChessSelfPlaySearch.inference_dimensions()
    module = StubNetwork(dimensions.actions, dimensions.channels, dimensions.rows, dimensions.columns, hidden)
    module.eval()
    torch.jit.save(torch.jit.script(module), str(destination))
    return destination


def search_parameters(arguments: LoopArguments) -> native.SelfPlaySearchParameters:
    tree = native.TreeSearchParameters(
        1.5,
        native.FirstPlayUrgencyParameters(native.FirstPlayUrgencyKind.REDUCED_PARENT_VALUE, 0.2),
        1.5,
        0.99,
        0.5,
    )
    return native.SelfPlaySearchParameters(arguments.baseline_visits, arguments.search_budget_blend, tree, 0.3, 0.25)


@dataclass
class ActiveGame:
    root: object
    ply: int


class LinearCongruentialRandom:
    """Deterministic source; keeps the harness free of a numpy dependency."""

    def __init__(self, seed: int) -> None:
        self.state = seed | 1

    def next_float(self) -> float:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (self.state >> 11) / float(1 << 53)

    def below(self, bound: int) -> int:
        return min(bound - 1, int(self.next_float() * bound))


def new_game(search: object, random: LinearCongruentialRandom, opening_plies: int) -> ActiveGame:
    while True:
        position = native.ChessPosition()
        for _ in range(random.below(opening_plies + 1)):
            actions = position.legal_actions()
            if not actions:
                break
            position = position.child(actions[random.below(len(actions))])
        if position.legal_actions():
            return ActiveGame(root=search.new_root(position), ply=0)


def select_action(visits: list, ply: int, greedy_after_ply: int, random: LinearCongruentialRandom) -> int:
    if ply >= greedy_after_ply:
        return min(visits, key=lambda visit: (-visit.visit_count, visit.action_id)).action_id
    target = random.next_float() * sum(visit.visit_count for visit in visits)
    accumulated = 0
    for visit in visits:
        accumulated += visit.visit_count
        if accumulated >= target:
            return visit.action_id
    return visits[-1].action_id


STATISTIC_FIELDS = (
    'treeSelectionNanoseconds',
    'boardEncodingNanoseconds',
    'resultProcessingNanoseconds',
    'treeBackupNanoseconds',
    'treeOwnerWaitNanoseconds',
    'inferenceNanoseconds',
    'evaluations',
    'modelInferenceCalls',
    'modelInferencePositions',
)


def statistics_delta(final: object, initial: object) -> dict[str, int]:
    return {name: getattr(final, name) - getattr(initial, name) for name in STATISTIC_FIELDS}


def run(arguments: LoopArguments, model_path: Path) -> dict:
    inference = native.BatchedInferenceParameters(
        arguments.inference_workers,
        arguments.inference_batch_size,
        arguments.outstanding_batches_per_worker,
    )
    device = native.InferenceDevice.CUDA if arguments.inference_device == 'cuda' else native.InferenceDevice.CPU
    backend = {
        'automatic': native.SdpaBackend.AUTOMATIC,
        'flash': native.SdpaBackend.FLASH,
        'memory_efficient': native.SdpaBackend.MEMORY_EFFICIENT,
        'math': native.SdpaBackend.MATH,
        'cudnn': native.SdpaBackend.CUDNN,
    }[arguments.sdpa_backend]
    runtime = native.InferenceConfiguration(arguments.device_id, str(model_path), device, backend)
    search = native.ChessSelfPlaySearch(runtime, search_parameters(arguments), inference, 0)
    random = LinearCongruentialRandom(arguments.seed)
    games = [new_game(search, random, arguments.opening_plies) for _ in range(arguments.games)]

    prepare_seconds = 0.0
    native_seconds = 0.0
    advance_seconds = 0.0
    visit_seconds = 0.0
    statistics_samples: list[tuple[float, float, float, float]] = []
    simulations = 0
    completed_games = 0
    initial_statistics = None
    measurement_started_at = time.perf_counter()

    for batch_index in range(arguments.warmup_batches + arguments.measured_batches):
        measuring = batch_index >= arguments.warmup_batches
        if measuring and initial_statistics is None:
            initial_statistics = search.inference_statistics()
            measurement_started_at = time.perf_counter()

        started = time.perf_counter()
        requests = []
        for game in games:
            game.root.discount(arguments.retained_root_visit_fraction)
            requests.append(search.request(game.root, parallel_searches=arguments.parallel_searches))
        prepared_at = time.perf_counter()
        batch = search.search(requests, arguments.collect_statistics)
        searched_at = time.perf_counter()

        for index, result in enumerate(batch.results):
            game = games[index]
            visit_started = time.perf_counter()
            visits = list(result.search_visits)
            policy_target = [(visit.action_id, visit.visit_count) for visit in result.policy_target_visits]
            visit_seconds += time.perf_counter() - visit_started
            observation = (
                game.ply,
                policy_target,
                result.root_value,
                result.highest_visited_child_action_id,
                result.highest_visited_child_visit_count,
                result.highest_visited_child_q,
                result.network_root_value,
                result.policy_correction,
                result.value_correction,
                result.search_budget_logit,
                result.predicted_search_budget,
                result.assigned_additional_visits,
                result.parallel_searches,
                result.spend_residual,
                result.starting_visits,
                result.final_visits,
                result.stop_reason,
            )
            del observation
            action_id = select_action(visits, game.ply, arguments.greedy_after_ply, random)
            game.root = result.root
            game.root.play(action_id)
            game.ply += 1
            if game.root.is_terminal or game.ply >= arguments.maximum_game_plies:
                games[index] = new_game(search, random, arguments.opening_plies)
                if measuring:
                    completed_games += 1
        finished = time.perf_counter()

        if measuring:
            statistics_samples.append(
                (
                    batch.statistics.average_depth,
                    batch.statistics.average_entropy,
                    batch.statistics.average_policy_search_kl_divergence,
                    sum(result.final_visits for result in batch.results) / len(batch.results),
                )
            )
            prepare_seconds += prepared_at - started
            native_seconds += searched_at - prepared_at
            advance_seconds += finished - searched_at
            simulations += batch.simulations_completed

    elapsed = time.perf_counter() - measurement_started_at
    delta = statistics_delta(search.inference_statistics(), initial_statistics)
    accounted = (
        delta['treeSelectionNanoseconds']
        + delta['boardEncodingNanoseconds']
        + delta['resultProcessingNanoseconds']
        + delta['treeBackupNanoseconds']
        + delta['treeOwnerWaitNanoseconds']
    )
    return {
        'configuration': {
            'games': arguments.games,
            'parallel_searches': arguments.parallel_searches,
            'baseline_visits': arguments.baseline_visits,
            'search_budget_blend': arguments.search_budget_blend,
            'inference_workers': arguments.inference_workers,
            'inference_batch_size': arguments.inference_batch_size,
            'inference_hidden': arguments.inference_hidden,
            'measured_batches': arguments.measured_batches,
        },
        'throughput': {
            'elapsed_seconds': elapsed,
            'simulations': simulations,
            'simulations_per_second': simulations / elapsed if elapsed else 0.0,
            'microseconds_per_simulation': 1e6 * elapsed / simulations if simulations else 0.0,
            'completed_games': completed_games,
        },
        'wall_split_seconds': {
            'python_prepare': prepare_seconds,
            'native_search': native_seconds,
            'python_advance': advance_seconds,
            'python_visit_materialization': visit_seconds,
        },
        'tree_thread_nanoseconds': {
            'tree_selection': delta['treeSelectionNanoseconds'],
            'board_encoding': delta['boardEncodingNanoseconds'],
            'result_processing': delta['resultProcessingNanoseconds'],
            'tree_backup': delta['treeBackupNanoseconds'],
            'inference_wait': delta['treeOwnerWaitNanoseconds'],
            'unaccounted': int(native_seconds * 1e9) - accounted,
        },
        'search_quality': {
            name: sum(sample[index] for sample in statistics_samples) / len(statistics_samples)
            for index, name in enumerate(
                ('average_depth', 'average_entropy', 'policy_search_kl', 'average_final_visits')
            )
        },
        'inference': {
            'thread_nanoseconds': delta['inferenceNanoseconds'],
            'model_calls': delta['modelInferenceCalls'],
            'model_positions': delta['modelInferencePositions'],
            'average_batch_size': (
                delta['modelInferencePositions'] / delta['modelInferenceCalls'] if delta['modelInferenceCalls'] else 0.0
            ),
        },
    }


def parse_arguments() -> tuple[LoopArguments, Path | None, str]:
    parser = argparse.ArgumentParser(
        description='Benchmark the native self-play search loop against a stand-in network.'
    )
    parser.add_argument('--games', type=int, default=128)
    parser.add_argument('--parallel-searches', type=int, default=2)
    parser.add_argument('--baseline-visits', type=int, default=250)
    parser.add_argument('--search-budget-blend', type=float, default=0.0)
    parser.add_argument('--retained-root-visit-fraction', type=float, default=0.6)
    parser.add_argument('--inference-workers', type=int, default=2)
    parser.add_argument('--inference-batch-size', type=int, default=256)
    parser.add_argument('--outstanding-batches-per-worker', type=int, default=2)
    parser.add_argument('--warmup-batches', type=int, default=3)
    parser.add_argument('--measured-batches', type=int, default=10)
    parser.add_argument('--maximum-game-plies', type=int, default=150)
    parser.add_argument('--opening-plies', type=int, default=12)
    parser.add_argument('--greedy-after-ply', type=int, default=60)
    parser.add_argument('--inference-hidden', type=int, default=0)
    parser.add_argument('--collect-statistics', action='store_true')
    parser.add_argument('--inference-device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument(
        '--sdpa-backend',
        choices=('automatic', 'flash', 'memory_efficient', 'math', 'cudnn'),
        default='memory_efficient',
    )
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--label', type=str, default='')
    namespace = parser.parse_args()
    arguments = LoopArguments(
        games=namespace.games,
        parallel_searches=namespace.parallel_searches,
        baseline_visits=namespace.baseline_visits,
        search_budget_blend=namespace.search_budget_blend,
        retained_root_visit_fraction=namespace.retained_root_visit_fraction,
        inference_workers=namespace.inference_workers,
        inference_batch_size=namespace.inference_batch_size,
        outstanding_batches_per_worker=namespace.outstanding_batches_per_worker,
        warmup_batches=namespace.warmup_batches,
        measured_batches=namespace.measured_batches,
        maximum_game_plies=namespace.maximum_game_plies,
        opening_plies=namespace.opening_plies,
        greedy_after_ply=namespace.greedy_after_ply,
        inference_hidden=namespace.inference_hidden,
        collect_statistics=namespace.collect_statistics,
        inference_device=namespace.inference_device,
        device_id=namespace.device_id,
        sdpa_backend=namespace.sdpa_backend,
        seed=namespace.seed,
    )
    return arguments, namespace.model, namespace.label


def main() -> None:
    torch.set_num_threads(1)
    arguments, model, label = parse_arguments()
    if model is None:
        directory = Path(os.environ.get('TMPDIR', '/tmp'))
        model = directory / f'self-play-stub-{arguments.inference_hidden}.jit.pt'
        write_stub_model(model, arguments.inference_hidden)
    result = run(arguments, model)
    result['label'] = label
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
