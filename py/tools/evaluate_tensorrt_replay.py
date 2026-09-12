from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import Field
from src.training.batch import TrainingModelOutput
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from tools.benchmark_tensorrt_inference import _TensorRtCudaGraphRunner
from tools.benchmark_training_overfit import LossValues
from tools.distill_train_student import (
    OpenedProductionReplay,
    ProductionReplayInput,
    close_training_dataset,
    dataset_split,
    distillation_objective,
    mean_loss_values,
    observed_losses,
    open_training_dataset,
)
from tools.run_int8_architecture_screen import _held_out_batches, _legal_action_mask, _onnx_outputs
from tools.tensorrt_benchmark_metrics import ModelOutputs, measure_fidelity

BATCH_SIZE = 320
POLICY_MARGIN_THRESHOLDS = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class Arguments:
    replay_store: Path
    replay_experiment: Path
    replay_sha256: str
    onnx: Path
    engine: Path
    output: Path
    device_id: int
    positions: int
    holdout_fraction: float


class MarginAgreement(FrozenModel):
    minimum_reference_margin: float = Field(ge=0.0)
    positions: int = Field(gt=0)
    agreement: float = Field(ge=0.0, le=1.0)


class TensorRtReplayReport(FrozenModel):
    schema_version: int = 1
    replay_store_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    onnx_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    engine_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    positions: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    tensorrt_loss: LossValues
    policy_margin_agreement_to_onnx: tuple[MarginAgreement, ...]


def _policy_margin_agreement(
    reference: ModelOutputs, candidate: ModelOutputs, legal_mask: torch.Tensor
) -> tuple[MarginAgreement, ...]:
    reference_logits = reference.policy_logits.masked_fill(~legal_mask, float('-inf'))
    candidate_logits = candidate.policy_logits.masked_fill(~legal_mask, float('-inf'))
    reference_top_two = torch.topk(reference_logits, k=2, dim=1)
    reference_margin = reference_top_two.values[:, 0] - reference_top_two.values[:, 1]
    agrees = reference_top_two.indices[:, 0] == candidate_logits.argmax(dim=1)
    results: list[MarginAgreement] = []
    for threshold in POLICY_MARGIN_THRESHOLDS:
        selected = reference_margin >= threshold
        if selected.any():
            results.append(
                MarginAgreement(
                    minimum_reference_margin=threshold,
                    positions=int(selected.sum()),
                    agreement=float(agrees[selected].float().mean()),
                )
            )
    return tuple(results)


def run(arguments: Arguments) -> TensorRtReplayReport:
    dataset_input = ProductionReplayInput(
        kind='production_replay',
        path=arguments.replay_store,
        experiment=arguments.replay_experiment,
        orchestrator_recorded_sha256=arguments.replay_sha256,
    )
    opened = open_training_dataset(dataset_input)
    if not isinstance(opened, OpenedProductionReplay):
        raise AssertionError('TensorRT replay evaluation requires a production replay store.')
    try:
        split = dataset_split(opened.row_count, arguments.holdout_fraction, 1.0)
        if arguments.positions > split.held_out_row_count:
            raise ValueError('Requested positions exceed the untouched holdout.')
        device = torch.device('cuda', arguments.device_id)
        batches = _held_out_batches(opened, split.held_out_start_row, arguments.positions, BATCH_SIZE, device)
        runner = _TensorRtCudaGraphRunner(arguments.engine, batches[0].states, device, 10)
        objective = distillation_objective()
        outputs: list[ModelOutputs] = []
        losses: list[LossValues] = []
        legal_masks: list[torch.Tensor] = []
        for batch in batches:
            runner.load_states(batch.states)
            output = runner.outputs()
            outputs.append(output)
            cpu_batch = batch.to_device(torch.device('cpu'), non_blocking=False)
            training_output = TrainingModelOutput(
                policy_logits=output.policy_logits,
                wdl_logits=output.wdl_probabilities.clamp_min(torch.finfo(torch.float32).tiny).log(),
                auxiliary_logits=(),
                features=torch.empty(0),
            )
            losses.append(observed_losses(objective.calculate_loss(training_output, cpu_batch)))
            legal_masks.append(_legal_action_mask(batch.policy_legal_action_ids))
        candidate = ModelOutputs(
            policy_logits=torch.cat(tuple(output.policy_logits for output in outputs)),
            wdl_probabilities=torch.cat(tuple(output.wdl_probabilities for output in outputs)),
        )
        legal_mask = torch.cat(tuple(legal_masks))
        reference = _onnx_outputs(arguments.onnx, batches, arguments.device_id)
        measure_fidelity(reference, candidate, legal_mask)
        report = TensorRtReplayReport(
            replay_store_sha256=arguments.replay_sha256,
            onnx_sha256=file_sha256(arguments.onnx),
            engine_sha256=file_sha256(arguments.engine),
            positions=arguments.positions,
            batch_size=BATCH_SIZE,
            tensorrt_loss=mean_loss_values(tuple(losses)),
            policy_margin_agreement_to_onnx=_policy_margin_agreement(reference, candidate, legal_mask),
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomically(arguments.output, report.model_dump_json(indent=2) + '\n')
        return report
    finally:
        close_training_dataset(opened)


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Evaluate a TensorRT engine against untouched replay targets.')
    parser.add_argument('--replay-store', required=True, type=Path)
    parser.add_argument('--replay-experiment', required=True, type=Path)
    parser.add_argument('--replay-sha256', required=True)
    parser.add_argument('--onnx', required=True, type=Path)
    parser.add_argument('--engine', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device-id', required=True, type=int)
    parser.add_argument('--positions', default=51_200, type=int)
    parser.add_argument('--holdout-fraction', default=0.02, type=float)
    namespace = parser.parse_args()
    if namespace.positions <= 0 or namespace.positions % BATCH_SIZE:
        raise ValueError(f'Positions must be a positive multiple of {BATCH_SIZE}.')
    return Arguments(
        replay_store=namespace.replay_store,
        replay_experiment=namespace.replay_experiment,
        replay_sha256=namespace.replay_sha256,
        onnx=namespace.onnx,
        engine=namespace.engine,
        output=namespace.output,
        device_id=namespace.device_id,
        positions=namespace.positions,
        holdout_fraction=namespace.holdout_fraction,
    )


def main() -> None:
    print(run(parse_arguments()).model_dump_json(indent=2))


if __name__ == '__main__':
    main()
