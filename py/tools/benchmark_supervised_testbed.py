from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from src.experiment.configuration import load_experiment_configuration
from src.games.composition import ConfiguredGame, create_game_implementation
from src.replay.batch_loader import MappedReplayBatchLoader, build_training_batch
from src.replay.description import ReplayDescription
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.training.batch import TrainingBatch
from src.training.checkpoint.persistence import create_model, create_optimizer
from src.training.configuration import TrainingPrecision
from src.training.network import (
    AttentionNetworkParams,
    DensePolicyHeadConfiguration,
    Network,
    NetworkConfiguration,
)
from src.training.objective import ResolvedTrainingObjective
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


class TestbedObservation(FrozenModel):
    optimizer_step: int
    elapsed_seconds: float
    training_total: float
    training_policy: float
    holdout_total: float
    holdout_policy: float
    holdout_wdl: float
    gradient_norm: float


class TestbedCellResult(FrozenModel):
    cell_id: str
    learning_rate: float
    warmup_optimizer_steps: int
    parameter_count: int
    completed_optimizer_steps: int
    elapsed_seconds: float
    optimizer_steps_per_second: float
    training_samples_per_second: float
    peak_allocated_mib: float
    initial_holdout_policy: float
    final_holdout_policy: float
    best_holdout_policy: float
    maximum_gradient_norm_after_200: float
    diverged: bool
    observations: tuple[TestbedObservation, ...]


class TestbedReport(FrozenModel):
    train_store: Path
    train_rows: int
    holdout_store: Path
    holdout_rows: int
    batch_size: int
    maximum_optimizer_steps: int
    time_budget_seconds: float
    random_seed: int
    results: tuple[TestbedCellResult, ...]


@dataclass(frozen=True)
class TestbedCell:
    cell_id: str
    network: NetworkConfiguration
    learning_rate: float
    warmup_optimizer_steps: int


@dataclass(frozen=True)
class Arguments:
    configuration: Path
    train_store: Path
    holdout_store: Path
    cells: tuple[str, ...]
    device_id: int
    batch_size: int
    holdout_rows: int
    report_interval: int
    maximum_optimizer_steps: int
    time_budget_seconds: float
    random_seed: int
    output_path: Path


def _attention_network(
    num_layers: int, embedding_size: int, num_heads: int, base: NetworkConfiguration
) -> NetworkConfiguration:
    return AttentionNetworkParams(
        num_layers=num_layers,
        embedding_size=embedding_size,
        num_heads=num_heads,
        feedforward_size=2 * embedding_size,
        dropout=0.0,
        policy_head=base.policy_head,
        num_value_channels=base.num_value_channels,
        value_fc_size=base.value_fc_size,
    )


def resolve_cell(cell_id: str, base_network: NetworkConfiguration) -> TestbedCell:
    match cell_id.split('@'):
        case ['dense-cnn', learning_rate]:
            network = base_network.validated_copy(
                update={'policy_head': DensePolicyHeadConfiguration(channels=4).model_dump(mode='json')}
            )
            return TestbedCell(cell_id, network, float(learning_rate), 0)
        case ['plane-cnn', learning_rate]:
            return TestbedCell(cell_id, base_network, float(learning_rate), 0)
        case ['attention-8x128', learning_rate]:
            return TestbedCell(cell_id, _attention_network(8, 128, 4, base_network), float(learning_rate), 1000)
        case ['attention-6x96', learning_rate]:
            return TestbedCell(cell_id, _attention_network(6, 96, 3, base_network), float(learning_rate), 1000)
        case _:
            raise ValueError(f'Unknown testbed cell: {cell_id}')


def _evaluate_holdout(
    model: Network,
    holdout_chunks: tuple[TrainingBatch, ...],
    objective: ResolvedTrainingObjective,
    precision: TrainingPrecision,
) -> tuple[float, float, float]:
    model.eval()
    totals = np.zeros(3)
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=holdout_chunks[0].states.device.type,
            dtype=torch.bfloat16,
            enabled=precision is TrainingPrecision.BFLOAT16,
        ),
    ):
        for chunk in holdout_chunks:
            loss = objective.calculate_loss(model.training_output(chunk.states), chunk)
            totals += (float(loss.total.detach()), float(loss.policy.detach()), float(loss.wdl.detach()))
    model.train()
    chunks = len(holdout_chunks)
    return totals[0] / chunks, totals[1] / chunks, totals[2] / chunks


def run_cell(
    arguments: Arguments,
    cell: TestbedCell,
    train_description: ReplayDescription,
    holdout_chunks: tuple[TrainingBatch, ...],
    objective: ResolvedTrainingObjective,
    precision: TrainingPrecision,
    game: ConfiguredGame,
) -> TestbedCellResult:
    torch.manual_seed(arguments.random_seed)
    torch.cuda.manual_seed_all(arguments.random_seed)
    device = torch.device('cuda', arguments.device_id)
    model = create_model(cell.network, device, game.network_dimensions, game.target_layout.auxiliary_heads)
    optimizer = create_optimizer(model, 'adamw')
    loader = MappedReplayBatchLoader(
        replay=train_description,
        state=game.state,
        source_optimizer_step=0,
        optimizer_steps=arguments.maximum_optimizer_steps,
        global_batch_size=arguments.batch_size,
        world_size=1,
        rank=0,
        sampler_seed=arguments.random_seed,
        pin_memory=True,
    )
    initial = _evaluate_holdout(model, holdout_chunks, objective, precision)
    observations: list[TestbedObservation] = []
    best_holdout_policy = initial[1]
    maximum_gradient_norm_after_200 = 0.0
    diverged = False
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    completed_steps = 0
    with loader.prefetch(device, uses_cuda=True, depth=4) as batches:
        for batch in batches:
            step = completed_steps + 1
            warmup_scale = min(1.0, step / cell.warmup_optimizer_steps) if cell.warmup_optimizer_steps else 1.0
            for parameter_group in optimizer.param_groups:
                parameter_group['lr'] = cell.learning_rate * warmup_scale
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision is TrainingPrecision.BFLOAT16,
            ):
                loss = objective.calculate_loss(model.training_output(batch.states), batch)
            loss.total.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5))
            optimizer.step()
            completed_steps = step
            if step > 200:
                maximum_gradient_norm_after_200 = max(maximum_gradient_norm_after_200, gradient_norm)
            elapsed = time.perf_counter() - started_at
            if step % arguments.report_interval == 0 or elapsed > arguments.time_budget_seconds:
                holdout_total, holdout_policy, holdout_wdl = _evaluate_holdout(
                    model, holdout_chunks, objective, precision
                )
                best_holdout_policy = min(best_holdout_policy, holdout_policy)
                observations.append(
                    TestbedObservation(
                        optimizer_step=step,
                        elapsed_seconds=elapsed,
                        training_total=float(loss.total.detach()),
                        training_policy=float(loss.policy.detach()),
                        holdout_total=holdout_total,
                        holdout_policy=holdout_policy,
                        holdout_wdl=holdout_wdl,
                        gradient_norm=gradient_norm,
                    )
                )
                if not np.isfinite(holdout_total) or not np.isfinite(float(loss.total.detach())):
                    diverged = True
                    break
            if elapsed > arguments.time_budget_seconds:
                break
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at
    final_holdout_policy = observations[-1].holdout_policy if observations else initial[1]
    result = TestbedCellResult(
        cell_id=cell.cell_id,
        learning_rate=cell.learning_rate,
        warmup_optimizer_steps=cell.warmup_optimizer_steps,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        completed_optimizer_steps=completed_steps,
        elapsed_seconds=elapsed_seconds,
        optimizer_steps_per_second=completed_steps / elapsed_seconds if elapsed_seconds else 0.0,
        training_samples_per_second=completed_steps * arguments.batch_size / elapsed_seconds
        if elapsed_seconds
        else 0.0,
        peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
        initial_holdout_policy=initial[1],
        final_holdout_policy=final_holdout_policy,
        best_holdout_policy=best_holdout_policy,
        maximum_gradient_norm_after_200=maximum_gradient_norm_after_200,
        diverged=diverged,
        observations=tuple(observations),
    )
    del optimizer
    del model
    torch.cuda.empty_cache()
    return result


def run_testbed(arguments: Arguments) -> TestbedReport:
    torch.cuda.set_device(arguments.device_id)
    configuration = load_experiment_configuration(arguments.configuration)
    game = create_game_implementation(configuration)
    replay_layout = ReplayLayout(
        packed_planes=game.state.packed_plane_layout,
        targets=game.target_layout,
        maximum_policy_entries=configuration.training.lifecycle.replay.maximum_policy_entries,
        maximum_legal_actions=game.state.maximum_legal_action_count,
    )
    objective = game.training_objective_at(0)
    precision = configuration.training.trainer.precision
    base_network = configuration.training.initial_model.network

    train_store = ReplayStore.open(arguments.train_store, replay_layout, writable=False)
    train_description = ReplayDescription(
        path=arguments.train_store,
        head=train_store.state.head,
        size=train_store.state.size,
        logical_capacity=train_store.state.logical_capacity,
        maximum_capacity=train_store.state.maximum_capacity,
        layout=replay_layout,
    )
    train_rows = train_store.state.size
    train_store.close()

    holdout_store = ReplayStore.open(arguments.holdout_store, replay_layout, writable=False)
    try:
        generator = np.random.default_rng(arguments.random_seed)
        holdout_indices = generator.choice(holdout_store.state.size, size=arguments.holdout_rows, replace=False)
        chunk_size = 1024
        holdout_chunks = tuple(
            build_training_batch(
                holdout_store,
                game.state,
                tuple(int(index) for index in holdout_indices[start : start + chunk_size]),
                (0,) * len(holdout_indices[start : start + chunk_size]),
            )
            for start in range(0, arguments.holdout_rows, chunk_size)
        )
        holdout_rows = holdout_store.state.size
    finally:
        holdout_store.close()
    device = torch.device('cuda', arguments.device_id)
    holdout_chunks = tuple(chunk.to_device(device, non_blocking=False) for chunk in holdout_chunks)

    results = []
    for cell_id in arguments.cells:
        cell = resolve_cell(cell_id, base_network)
        result = run_cell(arguments, cell, train_description, holdout_chunks, objective, precision, game)
        results.append(result)
        print(
            f'{result.cell_id}: steps={result.completed_optimizer_steps} '
            f'holdout_policy {result.initial_holdout_policy:.4f} -> {result.final_holdout_policy:.4f} '
            f'(best {result.best_holdout_policy:.4f}), grad_norm_max_200 {result.maximum_gradient_norm_after_200:.3f}, '
            f'{result.training_samples_per_second:.0f} samples/s, diverged={result.diverged}',
            flush=True,
        )
    game.close()
    report = TestbedReport(
        train_store=arguments.train_store,
        train_rows=train_rows,
        holdout_store=arguments.holdout_store,
        holdout_rows=holdout_rows,
        batch_size=arguments.batch_size,
        maximum_optimizer_steps=arguments.maximum_optimizer_steps,
        time_budget_seconds=arguments.time_budget_seconds,
        random_seed=arguments.random_seed,
        results=tuple(results),
    )
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomically(arguments.output_path, report.model_dump_json(indent=2) + '\n')
    return report


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Supervised testbed on a frozen replay store (recovery plan WP4).')
    parser.add_argument('--configuration', required=True, type=Path)
    parser.add_argument('--train-store', required=True, type=Path)
    parser.add_argument('--holdout-store', required=True, type=Path)
    parser.add_argument('--cells', required=True, nargs='+')
    parser.add_argument('--device-id', default=0, type=int)
    parser.add_argument('--batch-size', default=2048, type=int)
    parser.add_argument('--holdout-rows', default=4096, type=int)
    parser.add_argument('--report-interval', default=250, type=int)
    parser.add_argument('--maximum-optimizer-steps', default=4000, type=int)
    parser.add_argument('--time-budget-seconds', default=900.0, type=float)
    parser.add_argument('--random-seed', default=20260821, type=int)
    parser.add_argument('--output-path', required=True, type=Path)
    namespace = parser.parse_args()
    return Arguments(
        configuration=namespace.configuration,
        train_store=namespace.train_store,
        holdout_store=namespace.holdout_store,
        cells=tuple(namespace.cells),
        device_id=namespace.device_id,
        batch_size=namespace.batch_size,
        holdout_rows=namespace.holdout_rows,
        report_interval=namespace.report_interval,
        maximum_optimizer_steps=namespace.maximum_optimizer_steps,
        time_budget_seconds=namespace.time_budget_seconds,
        random_seed=namespace.random_seed,
        output_path=namespace.output_path,
    )


if __name__ == '__main__':
    run_testbed(parse_arguments())
