"""Measures whether a reduced-precision forward pass produces the same self-play targets.

Reduced precision is acceptable for data generation only if the priors and values the search
consumes are unchanged in the ways that matter. This compares every candidate precision and memory
format against the shipped bfloat16 contiguous path on stored positions and reports, per variant,
the top-1 policy agreement over legal moves, the policy KL divergence, and the value error.

The positions come from the evaluation dataset built by the run itself
(`chess-stockfish-evaluation-v1.bin` and its manifest), so the comparison runs on the position
distribution self-play actually visits rather than on synthetic planes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import Field
from src.evaluation.dataset import dataset_manifest_path, load_evaluation_dataset
from src.evaluation.inference import decode_packed_inputs
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.provenance import SourceRevision, read_source_revision
from torch import Tensor

REFERENCE_PRECISION = 'bfloat16'
REFERENCE_MEMORY_FORMAT = 'contiguous'


class Precision(str, Enum):
    BFLOAT16 = 'bfloat16'
    FLOAT16 = 'float16'
    FLOAT32 = 'float32'

    @property
    def torch_dtype(self) -> torch.dtype:
        match self:
            case Precision.BFLOAT16:
                return torch.bfloat16
            case Precision.FLOAT16:
                return torch.float16
            case Precision.FLOAT32:
                return torch.float32


class MemoryFormat(str, Enum):
    CONTIGUOUS = 'contiguous'
    CHANNELS_LAST = 'channels_last'

    @property
    def torch_memory_format(self) -> torch.memory_format:
        match self:
            case MemoryFormat.CONTIGUOUS:
                return torch.contiguous_format
            case MemoryFormat.CHANNELS_LAST:
                return torch.channels_last


@dataclass(frozen=True)
class AgreementArguments:
    model_path: Path
    dataset_path: Path
    output_path: Path
    gpu_id: int
    position_count: int
    batch_size: int
    acknowledge_gpu_load: bool


@dataclass(frozen=True)
class ModelOutputs:
    policy_logits: Tensor
    wdl_probabilities: Tensor
    search_corrections: Tensor


class VariantAgreement(FrozenModel):
    precision: Precision
    memory_format: MemoryFormat
    positions: int = Field(gt=0)
    legal_top1_agreement: float = Field(ge=0.0, le=1.0)
    unrestricted_top1_agreement: float = Field(ge=0.0, le=1.0)
    mean_policy_kl_divergence: float = Field(ge=0.0)
    maximum_policy_kl_divergence: float = Field(ge=0.0)
    value_mean_absolute_error: float = Field(ge=0.0)
    value_maximum_absolute_error: float = Field(ge=0.0)
    search_correction_mean_absolute_error: float = Field(ge=0.0)


class PrecisionAgreementReport(FrozenModel):
    schema_version: Literal[1] = 1
    source_revision: SourceRevision
    model_path: str = Field(min_length=1)
    dataset_path: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    reference_precision: str = Field(min_length=1)
    reference_memory_format: str = Field(min_length=1)
    variants: tuple[VariantAgreement, ...] = Field(min_length=1)


def masked_log_softmax(policy_logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Log-softmax restricted to legal actions, which is the distribution the search consumes."""
    masked = policy_logits.to(torch.float64).masked_fill(~legal_mask, float('-inf'))
    return torch.log_softmax(masked, dim=1)


def expected_value(wdl_probabilities: Tensor) -> Tensor:
    return wdl_probabilities.to(torch.float64)[:, 0] - wdl_probabilities.to(torch.float64)[:, 2]


def measure_agreement(
    reference: ModelOutputs,
    candidate: ModelOutputs,
    legal_mask: Tensor,
    precision: Precision,
    memory_format: MemoryFormat,
) -> VariantAgreement:
    reference_log_probabilities = masked_log_softmax(reference.policy_logits, legal_mask)
    candidate_log_probabilities = masked_log_softmax(candidate.policy_logits, legal_mask)
    reference_probabilities = reference_log_probabilities.exp()
    # KL(reference || candidate): the reference is the distribution the run is known to train on.
    divergence_terms = reference_probabilities * (reference_log_probabilities - candidate_log_probabilities)
    # A divergence is never negative; denormal reference probabilities can still sum to a value a
    # few ulps below zero, and clamping keeps that noise from reading as a negative divergence.
    kl_divergence = (
        torch.where(reference_probabilities > 0, divergence_terms, torch.zeros_like(divergence_terms))
        .sum(dim=1)
        .clamp_min(0.0)
    )

    legal_agreement = reference_log_probabilities.argmax(dim=1) == candidate_log_probabilities.argmax(dim=1)
    unrestricted_agreement = reference.policy_logits.argmax(dim=1) == candidate.policy_logits.argmax(dim=1)
    value_error = (expected_value(reference.wdl_probabilities) - expected_value(candidate.wdl_probabilities)).abs()
    correction_error = (
        reference.search_corrections.to(torch.float64) - candidate.search_corrections.to(torch.float64)
    ).abs()

    return VariantAgreement(
        precision=precision,
        memory_format=memory_format,
        positions=int(legal_mask.shape[0]),
        legal_top1_agreement=float(legal_agreement.to(torch.float64).mean()),
        unrestricted_top1_agreement=float(unrestricted_agreement.to(torch.float64).mean()),
        mean_policy_kl_divergence=float(kl_divergence.mean()),
        maximum_policy_kl_divergence=float(kl_divergence.max()),
        value_mean_absolute_error=float(value_error.mean()),
        value_maximum_absolute_error=float(value_error.max()),
        search_correction_mean_absolute_error=float(correction_error.mean()),
    )


def load_positions(dataset_path: Path, position_count: int) -> tuple[Tensor, Tensor]:
    """Returns the decoded network inputs and the legal-action mask for the stored positions."""
    from src.evaluation.contracts import EVALUATION_DATASET_MANIFEST_ADAPTER

    manifest_path = dataset_manifest_path(dataset_path)
    if not dataset_path.exists() or not manifest_path.exists():
        raise ValueError(f'The agreement harness needs the evaluation dataset and its manifest at {dataset_path}.')
    manifest = EVALUATION_DATASET_MANIFEST_ADAPTER.validate_json(manifest_path.read_text(encoding='utf-8'))
    if manifest.position_count < position_count:
        raise ValueError(
            f'The evaluation dataset holds {manifest.position_count} positions, '
            f'fewer than the {position_count} requested.'
        )
    rows = load_evaluation_dataset(dataset_path, manifest)[:position_count]
    packed_states = tuple(CHESS_STATE_CONTRACT.packed_plane_layout.value(bytes(row['packed_state'])) for row in rows)
    states = torch.from_numpy(decode_packed_inputs(CHESS_STATE_CONTRACT, packed_states))

    legal_mask = torch.zeros((len(rows), CHESS_STATE_CONTRACT.action_size), dtype=torch.bool)
    for index, row in enumerate(rows):
        legal_action_ids = np.asarray(row['legal_action_ids'][: int(row['legal_count'])], dtype=np.int64)
        legal_mask[index, torch.from_numpy(legal_action_ids)] = True
    return states, legal_mask


def run_variant(
    model_path: Path,
    states: Tensor,
    precision: Precision,
    memory_format: MemoryFormat,
    device: torch.device,
    batch_size: int,
) -> ModelOutputs:
    model = torch.jit.load(str(model_path), map_location=device)
    model.to(precision.torch_dtype)
    model.eval()
    if memory_format is MemoryFormat.CHANNELS_LAST:
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.dim() == 4:
                    parameter.set_data(parameter.data.contiguous(memory_format=torch.channels_last))
    frozen = torch.jit.freeze(model)

    policy_batches: list[Tensor] = []
    wdl_batches: list[Tensor] = []
    correction_batches: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, states.shape[0], batch_size):
            batch = (
                states[start : start + batch_size]
                .to(device=device, dtype=precision.torch_dtype)
                .to(memory_format=memory_format.torch_memory_format)
            )
            policy_logits, wdl_probabilities, search_corrections = frozen(batch)
            policy_batches.append(policy_logits.float().cpu())
            wdl_batches.append(wdl_probabilities.float().cpu())
            correction_batches.append(search_corrections.float().cpu())
    return ModelOutputs(
        policy_logits=torch.cat(policy_batches),
        wdl_probabilities=torch.cat(wdl_batches),
        search_corrections=torch.cat(correction_batches).flatten(),
    )


def run_agreement(arguments: AgreementArguments) -> PrecisionAgreementReport:
    if not arguments.acknowledge_gpu_load:
        raise ValueError('The agreement harness requires --acknowledge-gpu-load.')
    if not torch.cuda.is_available():
        raise ValueError('The agreement harness requires CUDA: the reference is the CUDA bfloat16 path.')
    if arguments.gpu_id >= torch.cuda.device_count():
        raise ValueError(f'GPU ID {arguments.gpu_id} is not available.')

    device = torch.device('cuda', arguments.gpu_id)
    torch.cuda.set_device(device)
    states, legal_mask = load_positions(arguments.dataset_path, arguments.position_count)

    reference = run_variant(
        arguments.model_path, states, Precision.BFLOAT16, MemoryFormat.CONTIGUOUS, device, arguments.batch_size
    )
    variants: list[VariantAgreement] = []
    for precision in Precision:
        for memory_format in MemoryFormat:
            candidate = run_variant(
                arguments.model_path, states, precision, memory_format, device, arguments.batch_size
            )
            variants.append(measure_agreement(reference, candidate, legal_mask, precision, memory_format))

    return PrecisionAgreementReport(
        source_revision=read_source_revision(),
        model_path=str(arguments.model_path),
        dataset_path=str(arguments.dataset_path),
        device_name=torch.cuda.get_device_properties(device).name,
        reference_precision=REFERENCE_PRECISION,
        reference_memory_format=REFERENCE_MEMORY_FORMAT,
        variants=tuple(variants),
    )


def parse_arguments() -> AgreementArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True, help='A TorchScript .jit.pt inference checkpoint.')
    parser.add_argument('--dataset-path', type=Path, required=True, help='chess-stockfish-evaluation-v1.bin')
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--position-count', type=int, default=480)
    parser.add_argument('--batch-size', type=int, default=320)
    parser.add_argument('--acknowledge-gpu-load', action='store_true')
    parsed = parser.parse_args()
    return AgreementArguments(
        model_path=parsed.model_path,
        dataset_path=parsed.dataset_path,
        output_path=parsed.output_path,
        gpu_id=parsed.gpu_id,
        position_count=parsed.position_count,
        batch_size=parsed.batch_size,
        acknowledge_gpu_load=parsed.acknowledge_gpu_load,
    )


def main() -> None:
    arguments = parse_arguments()
    report = run_agreement(arguments)
    write_text_atomically(arguments.output_path, report.model_dump_json(indent=2) + '\n')
    print(json.dumps({'device': report.device_name, 'positions': report.variants[0].positions}, indent=2))
    print(f'{"precision":10s} {"layout":14s} {"legal top-1":>12s} {"mean KL":>10s} {"max KL":>10s} {"value MAE":>10s}')
    for variant in report.variants:
        print(
            f'{variant.precision.value:10s} {variant.memory_format.value:14s} '
            f'{100 * variant.legal_top1_agreement:11.3f}% {variant.mean_policy_kl_divergence:10.3e} '
            f'{variant.maximum_policy_kl_divergence:10.3e} {variant.value_mean_absolute_error:10.3e}'
        )


if __name__ == '__main__':
    main()
