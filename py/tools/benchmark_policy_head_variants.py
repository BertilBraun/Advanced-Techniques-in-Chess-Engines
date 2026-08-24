from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from src.experiment.configuration import ExperimentConfiguration, load_experiment_configuration
from src.games.composition import ConfiguredGame, create_game_implementation
from src.replay.batch_loader import build_training_batch
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.training.batch import TrainingBatch
from src.training.checkpoint.persistence import create_model, create_optimizer
from src.training.network import (
    DensePolicyHeadConfiguration,
    GlobalPoolingResidualContext,
    Network,
    NetworkParams,
    ResidualContextPlacement,
)
from src.training.objective import mask_policy_logits
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from torch.nn import functional

DEFAULT_TRUNK_LAYERS = 12
DEFAULT_TRUNK_HIDDEN_SIZE = 128
HOLDOUT_CHUNK_ROWS = 1024

POLICY_HEAD_VARIANTS: tuple[tuple[str, DensePolicyHeadConfiguration], ...] = (
    ('a-ch4-baseline', DensePolicyHeadConfiguration(channels=4)),
    ('b-ch4-reduce1', DensePolicyHeadConfiguration(channels=4, spatial_reductions=1)),
    ('c-ch8-reduce2', DensePolicyHeadConfiguration(channels=8, spatial_reductions=2)),
    ('d-ch4-rank64', DensePolicyHeadConfiguration(channels=4, bottleneck_rank=64)),
    ('e-ch4-rank96', DensePolicyHeadConfiguration(channels=4, bottleneck_rank=96)),
    ('f-ch8-reduce1-rank64', DensePolicyHeadConfiguration(channels=8, spatial_reductions=1, bottleneck_rank=64)),
    ('g-ch8-reduce1-rank96', DensePolicyHeadConfiguration(channels=8, spatial_reductions=1, bottleneck_rank=96)),
)


class VariantResult(FrozenModel):
    variant_id: str
    policy_head: DensePolicyHeadConfiguration
    head_parameter_count: int
    total_parameter_count: int
    completed_steps: int
    initial_holdout_policy_cross_entropy: float
    best_holdout_policy_cross_entropy: float
    final_holdout_policy_cross_entropy: float
    wall_seconds: float
    samples_per_second: float


class BakeOffReport(FrozenModel):
    store: Path
    configuration: Path
    device: str
    trunk_layers: int
    trunk_hidden_size: int
    action_size: int
    store_rows: int
    training_rows: int
    holdout_rows: int
    steps: int
    batch_size: int
    evaluation_interval: int
    learning_rate: float
    random_seed: int
    results: tuple[VariantResult, ...]


@dataclass(frozen=True)
class Arguments:
    store: Path
    configuration: Path
    device_id: int
    steps: int
    batch_size: int
    holdout_rows: int
    evaluation_interval: int
    learning_rate: float
    trunk_layers: int
    trunk_hidden_size: int
    variants: tuple[str, ...]
    random_seed: int
    output_path: Path


@dataclass(frozen=True)
class RowSplit:
    training: npt.NDArray[np.int64]
    holdout: npt.NDArray[np.int64]


def resolve_device(device_id: int) -> torch.device:
    return torch.device('cuda', device_id) if torch.cuda.is_available() else torch.device('cpu')


def resolve_variants(selected: tuple[str, ...]) -> tuple[tuple[str, DensePolicyHeadConfiguration], ...]:
    available = dict(POLICY_HEAD_VARIANTS)
    unknown = tuple(variant_id for variant_id in selected if variant_id not in available)
    if unknown:
        raise ValueError(f'Unknown policy head variants: {", ".join(unknown)}.')
    if not selected:
        return POLICY_HEAD_VARIANTS
    return tuple((variant_id, available[variant_id]) for variant_id in selected)


def split_rows(row_count: int, holdout_rows: int, random_seed: int) -> RowSplit:
    if holdout_rows <= 0 or holdout_rows >= row_count:
        raise ValueError('The holdout must be a nonempty proper subset of the store rows.')
    shuffled = np.random.default_rng(random_seed).permutation(row_count).astype(np.int64)
    return RowSplit(training=np.sort(shuffled[holdout_rows:]), holdout=np.sort(shuffled[:holdout_rows]))


def replay_layout_for(configuration: ExperimentConfiguration, game: ConfiguredGame) -> ReplayLayout:
    return ReplayLayout(
        packed_planes=game.state.packed_plane_layout,
        targets=game.target_layout,
        maximum_policy_entries=configuration.training.lifecycle.replay.maximum_policy_entries,
        maximum_legal_actions=game.state.maximum_legal_action_count,
    )


def trunk_configuration(
    arguments: Arguments,
    policy_head: DensePolicyHeadConfiguration,
) -> NetworkParams:
    return NetworkParams(
        num_layers=arguments.trunk_layers,
        hidden_size=arguments.trunk_hidden_size,
        residual_context=GlobalPoolingResidualContext(placement=ResidualContextPlacement.EVERY_SECOND_BLOCK),
        policy_head=policy_head,
    )


def policy_cross_entropy(model: Network, batch: TrainingBatch) -> torch.Tensor:
    logits = model.training_output(batch.states).policy_logits
    return functional.cross_entropy(mask_policy_logits(logits, batch.policy_legal_action_ids), batch.policy_targets)


def evaluate_holdout(model: Network, holdout_chunks: tuple[TrainingBatch, ...]) -> float:
    model.eval()
    with torch.no_grad():
        total = sum(float(policy_cross_entropy(model, chunk).detach()) for chunk in holdout_chunks)
    model.train()
    return total / len(holdout_chunks)


def build_holdout_chunks(
    store: ReplayStore,
    game: ConfiguredGame,
    holdout_indices: npt.NDArray[np.int64],
    device: torch.device,
) -> tuple[TrainingBatch, ...]:
    chunks = tuple(
        build_training_batch(
            store,
            game.state,
            holdout_indices[start : start + HOLDOUT_CHUNK_ROWS],
            np.zeros(len(holdout_indices[start : start + HOLDOUT_CHUNK_ROWS]), dtype=np.int64),
        )
        for start in range(0, len(holdout_indices), HOLDOUT_CHUNK_ROWS)
    )
    return tuple(chunk.to_device(device, non_blocking=False) for chunk in chunks)


def run_variant(
    arguments: Arguments,
    variant_id: str,
    policy_head: DensePolicyHeadConfiguration,
    store: ReplayStore,
    game: ConfiguredGame,
    training_indices: npt.NDArray[np.int64],
    holdout_chunks: tuple[TrainingBatch, ...],
    device: torch.device,
) -> VariantResult:
    torch.manual_seed(arguments.random_seed)
    torch.cuda.manual_seed_all(arguments.random_seed)
    model = create_model(trunk_configuration(arguments, policy_head), device, game.network_dimensions)
    optimizer = create_optimizer(model, 'adamw')
    for parameter_group in optimizer.param_groups:
        parameter_group['lr'] = arguments.learning_rate
    initial = evaluate_holdout(model, holdout_chunks)
    best = initial
    generator = np.random.default_rng(arguments.random_seed)
    started_at = time.perf_counter()
    evaluation_seconds = 0.0
    completed_steps = 0
    for step in range(1, arguments.steps + 1):
        sampled = generator.choice(training_indices, size=arguments.batch_size, replace=False)
        augmentations = generator.integers(0, game.state.augmentation_count, size=arguments.batch_size)
        batch = build_training_batch(store, game.state, sampled, augmentations.astype(np.int64)).to_device(
            device, non_blocking=False
        )
        optimizer.zero_grad(set_to_none=True)
        loss = policy_cross_entropy(model, batch)
        loss.backward()
        optimizer.step()
        completed_steps = step
        if step % arguments.evaluation_interval == 0 and step != arguments.steps:
            evaluation_started_at = time.perf_counter()
            best = min(best, evaluate_holdout(model, holdout_chunks))
            evaluation_seconds += time.perf_counter() - evaluation_started_at
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started_at - evaluation_seconds
    final = evaluate_holdout(model, holdout_chunks)
    best = min(best, final)
    return VariantResult(
        variant_id=variant_id,
        policy_head=policy_head,
        head_parameter_count=sum(parameter.numel() for parameter in model.policy_head.parameters()),
        total_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        completed_steps=completed_steps,
        initial_holdout_policy_cross_entropy=initial,
        best_holdout_policy_cross_entropy=best,
        final_holdout_policy_cross_entropy=final,
        wall_seconds=wall_seconds,
        samples_per_second=completed_steps * arguments.batch_size / wall_seconds if wall_seconds else 0.0,
    )


def run_bake_off(arguments: Arguments) -> BakeOffReport:
    variants = resolve_variants(arguments.variants)
    device = resolve_device(arguments.device_id)
    configuration = load_experiment_configuration(arguments.configuration)
    game = create_game_implementation(configuration)
    layout = replay_layout_for(configuration, game)
    store = ReplayStore.open(arguments.store, layout, writable=False)
    try:
        row_count = store.state.size
        split = split_rows(row_count, arguments.holdout_rows, arguments.random_seed)
        holdout_chunks = build_holdout_chunks(store, game, split.holdout, device)
        results = []
        for variant_id, policy_head in variants:
            result = run_variant(
                arguments,
                variant_id,
                policy_head,
                store,
                game,
                split.training,
                holdout_chunks,
                device,
            )
            results.append(result)
            print(
                f'{result.variant_id}: head={result.head_parameter_count} total={result.total_parameter_count} '
                f'holdout_policy {result.initial_holdout_policy_cross_entropy:.4f} -> '
                f'{result.final_holdout_policy_cross_entropy:.4f} '
                f'(best {result.best_holdout_policy_cross_entropy:.4f}), '
                f'{result.samples_per_second:.0f} samples/s',
                flush=True,
            )
    finally:
        store.close()
        game.close()
    report = BakeOffReport(
        store=arguments.store,
        configuration=arguments.configuration,
        device=str(device),
        trunk_layers=arguments.trunk_layers,
        trunk_hidden_size=arguments.trunk_hidden_size,
        action_size=game.network_dimensions.actions,
        store_rows=row_count,
        training_rows=len(split.training),
        holdout_rows=len(split.holdout),
        steps=arguments.steps,
        batch_size=arguments.batch_size,
        evaluation_interval=arguments.evaluation_interval,
        learning_rate=arguments.learning_rate,
        random_seed=arguments.random_seed,
        results=tuple(results),
    )
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomically(arguments.output_path, report.model_dump_json(indent=2) + '\n')
    return report


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Supervised policy-only bake-off across dense policy head variants.')
    parser.add_argument('--store', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--steps', default=2000, type=int)
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--holdout-rows', default=8192, type=int)
    parser.add_argument('--evaluation-interval', default=250, type=int)
    parser.add_argument('--learning-rate', default=0.002, type=float)
    parser.add_argument('--trunk-layers', default=DEFAULT_TRUNK_LAYERS, type=int)
    parser.add_argument('--trunk-hidden-size', default=DEFAULT_TRUNK_HIDDEN_SIZE, type=int)
    parser.add_argument('--variants', nargs='*', default=[])
    parser.add_argument('--random-seed', default=20260824, type=int)
    parser.add_argument('--output', required=True, type=Path)
    namespace = parser.parse_args()
    return Arguments(
        store=namespace.store,
        configuration=namespace.config,
        device_id=namespace.device,
        steps=namespace.steps,
        batch_size=namespace.batch_size,
        holdout_rows=namespace.holdout_rows,
        evaluation_interval=namespace.evaluation_interval,
        learning_rate=namespace.learning_rate,
        trunk_layers=namespace.trunk_layers,
        trunk_hidden_size=namespace.trunk_hidden_size,
        variants=tuple(namespace.variants),
        random_seed=namespace.random_seed,
        output_path=namespace.output,
    )


if __name__ == '__main__':
    run_bake_off(parse_arguments())
