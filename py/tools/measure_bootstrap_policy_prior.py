from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from src.distillation.dataset import build_training_batch, open_dataset
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS, CHESS_STATE_CONTRACT
from src.training.checkpoint.contracts import read_checkpoint_manifest
from src.training.checkpoint.persistence import create_model, create_optimizer, save_model_and_optimizer
from src.training.model_cost import measure_model_cost
from src.training.network import (
    POLICY_PRIOR_PROBE_POSITIONS,
    InferenceNetwork,
    Network,
    calibrate_bootstrap_policy_prior,
    measure_policy_prior_shape,
)
from src.util.log import log
from tools.attention_viability_cells import ARCHITECTURE_CELLS
from tools.distill_train_student import student_architecture

# The visit distribution PUCT produces at the root is driven by the prior, so a prior this flat leaves
# every legal move within a rounding error of every other one and search never breaks symmetry.
NEAR_UNIFORM_TOP3_MASS = 0.15


@dataclass(frozen=True)
class BootstrapPriorMeasurement:
    cell: str
    trunk: str
    total_parameters: int
    logit_standard_deviation: float
    uncalibrated_top1_mass: float
    uncalibrated_top3_mass: float
    applied_scale: float
    calibrated_top1_mass: float
    calibrated_top3_mass: float
    manifest_records_calibration: bool


def probe_states(dataset: Path, position_count: int, device: torch.device) -> torch.Tensor:
    records, manifest = open_dataset(dataset)
    rows = records[len(records) - position_count :]
    return build_training_batch(rows, CHESS_STATE_CONTRACT, manifest.action_size, device).states


def exported_model(model: Network) -> InferenceNetwork:
    export = InferenceNetwork(model)
    export.eval()
    export.fuse_model()
    return export


def policy_logit_standard_deviation(export: InferenceNetwork, states: torch.Tensor) -> float:
    with torch.no_grad():
        policy_logits, _, _ = export(states)
    return float(policy_logits.double().std())


def measure_cell(name: str, architecture, states: torch.Tensor, save_folder: Path) -> BootstrapPriorMeasurement:
    device = states.device
    model = create_model(architecture, device, CHESS_NETWORK_DIMENSIONS)
    export = exported_model(model)
    uncalibrated = measure_policy_prior_shape(export, states)
    logit_standard_deviation = policy_logit_standard_deviation(export, states)
    calibration = calibrate_bootstrap_policy_prior(export, states)

    save_folder.mkdir(parents=True, exist_ok=True)
    save_model_and_optimizer(model, create_optimizer(model, 'adamw'), 0, save_folder, states)
    manifest = read_checkpoint_manifest(0, save_folder)

    return BootstrapPriorMeasurement(
        cell=name,
        trunk=architecture.kind,
        total_parameters=measure_model_cost(model).parameters.total,
        logit_standard_deviation=logit_standard_deviation,
        uncalibrated_top1_mass=uncalibrated.top1_mass,
        uncalibrated_top3_mass=uncalibrated.top3_mass,
        applied_scale=calibration.applied_scale,
        calibrated_top1_mass=calibration.calibrated_shape.top1_mass,
        calibrated_top3_mass=calibration.calibrated_shape.top3_mass,
        manifest_records_calibration=manifest.policy_prior_calibration is not None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Measure the generation-0 exported policy prior of every architecture cell on real positions.'
    )
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scratch', required=True, type=Path)
    parser.add_argument('--positions', default=POLICY_PRIOR_PROBE_POSITIONS, type=int)
    parser.add_argument('--random-seed', default=20260827, type=int)
    namespace = parser.parse_args()

    torch.manual_seed(namespace.random_seed)
    device = torch.device('cpu')
    states = probe_states(namespace.dataset, namespace.positions, device)
    log(f'Probing with {len(states)} real positions from {namespace.dataset}.')

    measurements = []
    for cell in ARCHITECTURE_CELLS:
        measurement = measure_cell(
            cell.name,
            student_architecture(cell.arguments),
            states,
            namespace.scratch / cell.name,
        )
        measurements.append(measurement)
        log(
            f'{measurement.cell} ({measurement.trunk}): logit std '
            f'{measurement.logit_standard_deviation:.4g}, top-3 mass '
            f'{measurement.uncalibrated_top3_mass:.4f} -> {measurement.calibrated_top3_mass:.4f} at scale '
            f'{measurement.applied_scale:.4g}, manifest records the calibration: '
            f'{measurement.manifest_records_calibration}.'
        )

    namespace.output.parent.mkdir(parents=True, exist_ok=True)
    namespace.output.write_text(
        json.dumps(
            {
                'probe_positions': len(states),
                'near_uniform_top3_mass': NEAR_UNIFORM_TOP3_MASS,
                'measurements': [asdict(measurement) for measurement in measurements],
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    log(f'Wrote {namespace.output}.')


if __name__ == '__main__':
    main()
