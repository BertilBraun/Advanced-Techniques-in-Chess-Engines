from __future__ import annotations

import argparse
from enum import StrEnum
from pathlib import Path

import torch
from src.evaluation.dataset import load_dataset_probe_states
from src.evaluation.process import resolve_project_path
from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.composition import create_game_implementation
from src.training.bootstrap import select_bootstrap_model
from src.training.quantization.configuration import TensorRtInt8QatConfiguration
from src.training.quantization.runtime import (
    configure_qat,
    export_float_qat_onnx,
    export_qat_onnx,
    fixed_batch_example_states,
    fold_scaled_post_activation_batch_norm,
)


class TemplateExportKind(StrEnum):
    FLOAT_PRE_FOLD = 'float_pre_fold'
    FLOAT_DEPLOYMENT = 'float_deployment'
    INT8_PRE_FOLD = 'int8_pre_fold'
    INT8_DEPLOYMENT = 'int8_deployment'


def export_template(
    configuration_path: Path,
    model_id: str,
    export_kind: TemplateExportKind,
    output_path: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    configuration = load_experiment_configuration(configuration_path)
    if not isinstance(configuration, ChessExperimentConfiguration):
        raise ValueError('QAT TensorRT template export currently supports chess experiments.')
    quantization = configuration.training.trainer.quantization
    if not isinstance(quantization, TensorRtInt8QatConfiguration):
        raise ValueError('The experiment must configure TensorRT INT8 QAT.')
    matching_models = tuple(
        model for model in configuration.training.progressive_model_sizing.models if model.model_id == model_id
    )
    if len(matching_models) != 1:
        raise ValueError(f'Model ID must identify exactly one configured model: {model_id}')

    game = create_game_implementation(configuration)
    probe_states = load_dataset_probe_states(
        resolve_project_path(configuration.evaluation.dataset.path),
        game.state,
        configuration.training.trainer.bootstrap_probe_positions,
    ).to(device=device, dtype=torch.float32)
    selected = select_bootstrap_model(
        matching_models[0].network,
        device,
        game.network_dimensions,
        game.target_layout.auxiliary_heads,
        probe_states,
        configuration.training.trainer.bootstrap_initialization,
        configuration.training.trainer.bootstrap_policy_prior_target_top3_mass,
        configuration.training.random_seed,
    )
    model = selected.model
    calibration_states = load_dataset_probe_states(
        resolve_project_path(configuration.evaluation.dataset.path),
        game.state,
        quantization.calibration_positions,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    def calibrate(candidate: torch.nn.Module) -> None:
        was_training = candidate.training
        candidate.eval()
        with torch.inference_mode():
            for states in calibration_states.split(configuration.chess.self_play.inference.inference_batch_size):
                candidate(states)
        candidate.train(was_training)

    model = configure_qat(model, calibrate)
    if export_kind in (TemplateExportKind.FLOAT_DEPLOYMENT, TemplateExportKind.INT8_DEPLOYMENT):
        fold_scaled_post_activation_batch_norm(model)
    example_states = fixed_batch_example_states(calibration_states, batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if export_kind in (TemplateExportKind.FLOAT_PRE_FOLD, TemplateExportKind.FLOAT_DEPLOYMENT):
        export_float_qat_onnx(
            model,
            output_path,
            example_states,
            constant_folding=True,
        )
    else:
        export_qat_onnx(model, output_path, example_states)


def main() -> None:
    parser = argparse.ArgumentParser(description='Export a deterministic QAT ONNX graph for TensorRT templating.')
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--kind', type=TemplateExportKind, choices=tuple(TemplateExportKind), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    arguments = parser.parse_args()
    export_template(
        arguments.configuration,
        arguments.model_id,
        arguments.kind,
        arguments.output,
        arguments.batch_size,
        torch.device(arguments.device),
    )


if __name__ == '__main__':
    main()
