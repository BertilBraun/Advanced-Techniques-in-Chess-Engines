from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from src.distillation.dataset import build_training_batch, open_dataset
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS, CHESS_STATE_CONTRACT
from src.training.batch import TrainingBatch
from src.training.checkpoint.paths import model_save_path
from src.training.checkpoint.persistence import load_model
from src.training.network import Network
from src.training.objective import mask_policy_logits
from src.util.log import log
from tools.attention_viability_cells import ARCHITECTURE_CELLS, cell_by_name
from tools.distill_train_student import student_architecture

EVALUATION_BATCH_SIZE = 512
BOOTSTRAP_CHUNK = 100


@dataclass(frozen=True)
class PairedDifference:
    cell: str
    reference: str
    mean_difference: float
    confidence_low: float
    confidence_high: float
    positions: int


@dataclass(frozen=True)
class CellCrossEntropy:
    cell: str
    mean_policy_cross_entropy: float
    mean_gap_above_floor: float
    positions: int


def held_out_batches(dataset: Path, position_count: int, device: torch.device) -> tuple[tuple[TrainingBatch, ...], int]:
    records, manifest = open_dataset(dataset)
    start = len(records) - position_count
    batches = tuple(
        build_training_batch(
            records[start + offset : start + min(offset + EVALUATION_BATCH_SIZE, position_count)],
            CHESS_STATE_CONTRACT,
            manifest.action_size,
            device,
        )
        for offset in range(0, position_count, EVALUATION_BATCH_SIZE)
    )
    return batches, manifest.action_size


def target_entropy(targets: torch.Tensor) -> torch.Tensor:
    return -(targets * targets.clamp_min(torch.finfo(targets.dtype).tiny).log()).sum(dim=1)


def per_position_gap(model: Network, batches: tuple[TrainingBatch, ...], device: torch.device) -> np.ndarray:
    model.eval()
    gaps: list[np.ndarray] = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for batch in batches:
            policy_logits, _ = model.logit_forward(batch.states)
            masked = mask_policy_logits(policy_logits.float(), batch.policy_legal_action_ids)
            cross_entropy = torch.nn.functional.cross_entropy(masked, batch.policy_targets, reduction='none')
            gaps.append((cross_entropy - target_entropy(batch.policy_targets)).double().cpu().numpy())
    return np.concatenate(gaps)


def paired_bootstrap(differences: np.ndarray, samples: int, generator: np.random.Generator) -> tuple[float, float]:
    # Resampling all of them at once would allocate samples x positions indices, so they are drawn in chunks.
    means = np.empty(samples)
    for start in range(0, samples, BOOTSTRAP_CHUNK):
        stop = min(start + BOOTSTRAP_CHUNK, samples)
        indices = generator.integers(0, len(differences), size=(stop - start, len(differences)))
        means[start:stop] = differences[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_cell_model(cell_name: str, run_state: Path, generation: int, device: torch.device) -> Network:
    cell = cell_by_name(cell_name)
    return load_model(
        model_save_path(generation, run_state),
        student_architecture(cell.arguments),
        device,
        CHESS_NETWORK_DIMENSIONS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Paired held-out policy cross-entropy differences between the architecture cells.'
    )
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--run-state-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', default='cnn-A')
    parser.add_argument('--generation', default=322, type=int)
    parser.add_argument('--positions', default=32_768, type=int)
    parser.add_argument('--bootstrap-samples', default=10_000, type=int)
    parser.add_argument('--device-id', default=0, type=int)
    parser.add_argument('--random-seed', default=20260827, type=int)
    namespace = parser.parse_args()

    device = torch.device('cuda', namespace.device_id) if torch.cuda.is_available() else torch.device('cpu')
    batches, _ = held_out_batches(namespace.dataset, namespace.positions, device)
    log(f'Evaluating {namespace.positions} held-out positions in {len(batches)} batches on {device}.')

    floor = np.concatenate([target_entropy(batch.policy_targets).double().cpu().numpy() for batch in batches])
    log(f'Held-out policy floor over these positions: {floor.mean():.4f} nats.')

    gaps: dict[str, np.ndarray] = {}
    summaries: list[CellCrossEntropy] = []
    for cell in ARCHITECTURE_CELLS:
        run_state = namespace.run_state_root / cell.name
        if not model_save_path(namespace.generation, run_state).is_file():
            log(f'{run_state} carries no generation {namespace.generation} model; skipping {cell.name}.')
            continue
        model = load_cell_model(cell.name, run_state, namespace.generation, device)
        gap = per_position_gap(model, batches, device)
        gaps[cell.name] = gap
        summaries.append(
            CellCrossEntropy(
                cell=cell.name,
                mean_policy_cross_entropy=float((gap + floor).mean()),
                mean_gap_above_floor=float(gap.mean()),
                positions=len(gap),
            )
        )
        log(f'{cell.name}: mean gap above floor {gap.mean():.4f} over {len(gap)} positions.')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    generator = np.random.default_rng(namespace.random_seed)
    differences: list[PairedDifference] = []
    if namespace.reference in gaps:
        for name, gap in gaps.items():
            if name == namespace.reference:
                continue
            paired = gap - gaps[namespace.reference]
            low, high = paired_bootstrap(paired, namespace.bootstrap_samples, generator)
            differences.append(
                PairedDifference(
                    cell=name,
                    reference=namespace.reference,
                    mean_difference=float(paired.mean()),
                    confidence_low=low,
                    confidence_high=high,
                    positions=len(paired),
                )
            )
            log(f'{name} minus {namespace.reference}: {paired.mean():+.4f} nats [{low:+.4f}, {high:+.4f}] (95%)')

    namespace.output.parent.mkdir(parents=True, exist_ok=True)
    namespace.output.write_text(
        json.dumps(
            {
                'positions': namespace.positions,
                'bootstrap_samples': namespace.bootstrap_samples,
                'cells': [asdict(summary) for summary in summaries],
                'paired_differences': [asdict(difference) for difference in differences],
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    log(f'Wrote {namespace.output}.')


if __name__ == '__main__':
    main()
