"""Wraps an Lc0 ONNX network as a TorchScript model this project's inference pipeline can serve.

The pipeline hands the model an int8 (batch, 112, 8, 8) tensor and requires back policy logits over
1880 actions, finite for every legal move, and a WDL triple that is already a probability
distribution. This tool bakes the policy permutation and those conversions into a scripted module so
the C++ side needs no Lc0 awareness at all, and swapping to a larger Lc0 network is a re-export
rather than a code change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

LC0_POLICY_SIZE = 1858
PROJECT_ACTION_SIZE = 1880
LC0_INPUT_PLANES = 112
WDL_SIZE = 3
# The largest batch the evaluation pipeline sends (--inference-batch-size); larger callers chunk.
FIXED_BATCH = 64
# Finite, because the pipeline rejects a non-finite logit on any action it considers legal.
UNMAPPED_LOGIT = -1.0e4


class Lc0TeacherModel(nn.Module):
    """Adapts Lc0's policy indexing and value head to this project's contract.

    The planes reach the network exactly as Lc0's own encoder produces them: for the classical input
    the rule-50 plane carries the raw ply count, which is what the exported graph expects.
    """

    def __init__(
        self,
        backbone: nn.Module,
        permutation: torch.Tensor,
        wdl_is_already_probability: bool,
        policy_output_index: int,
        wdl_output_index: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer('permutation', permutation)
        self.register_buffer('padding', torch.zeros((FIXED_BATCH, LC0_INPUT_PLANES, 8, 8)))
        self.wdl_is_already_probability = wdl_is_already_probability
        self.policy_output_index = policy_output_index
        self.wdl_output_index = wdl_output_index

    def forward(self, encoded_boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The pipeline hands the network its own dtype (bfloat16 in evaluation), so no cast here. The
        # ONNX conversion's reshapes freeze whatever batch size they were traced with, so the backbone
        # always sees exactly FIXED_BATCH rows: pad up, run, and keep the real rows.
        planes = encoded_boards[:, :LC0_INPUT_PLANES]
        batch = planes.shape[0]
        padded = torch.cat((planes, self.padding), dim=0)[:FIXED_BATCH]
        outputs = [output[:batch] for output in self.backbone(padded)]
        lc0_policy = outputs[self.policy_output_index].to(torch.float32)
        lc0_wdl = outputs[self.wdl_output_index].to(torch.float32)
        # Derived from the output rather than torch.full: tracing bakes an explicit device and batch size
        # into a created tensor, and the first gate run failed on exactly that CPU/CUDA mismatch.
        unmapped = lc0_policy[:, :1] * 0.0 + UNMAPPED_LOGIT
        policy = torch.cat((lc0_policy, unmapped), dim=1).index_select(1, self.permutation)
        wdl = lc0_wdl if self.wdl_is_already_probability else torch.softmax(lc0_wdl, dim=1)
        return policy, wdl


def locate_outputs(backbone: nn.Module) -> tuple[int, int]:
    """Exports differ in output order and may carry a moves-left head, so outputs are found by width."""
    with torch.inference_mode():
        probe = torch.zeros((2, LC0_INPUT_PLANES, 8, 8), dtype=torch.float32)
        probe[:, LC0_INPUT_PLANES - 1] = 1.0
        outputs = backbone(probe)
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    widths = [tuple(output.shape) for output in outputs]
    print(f'Backbone outputs: {widths}')
    policy = [index for index, shape in enumerate(widths) if shape[-1] == LC0_POLICY_SIZE]
    wdl = [index for index, shape in enumerate(widths) if shape[-1] == WDL_SIZE]
    if len(policy) != 1 or len(wdl) != 1:
        raise SystemExit(f'Expected one {LC0_POLICY_SIZE}-wide policy and one {WDL_SIZE}-wide WDL output: {widths}')
    return policy[0], wdl[0]


def load_backbone(onnx_path: Path) -> nn.Module:
    try:
        from onnx2torch import convert
    except ImportError as error:
        raise SystemExit('onnx2torch is required to convert an Lc0 ONNX export.') from error
    return convert(str(onnx_path)).eval()


def build_permutation(policy_map_path: Path) -> torch.Tensor:
    """For each project action, the Lc0 index whose logit it takes; LC0_POLICY_SIZE marks unmapped."""
    payload = json.loads(policy_map_path.read_text(encoding='utf-8'))
    entries = payload['action_id_to_lc0_index']
    if len(entries) != PROJECT_ACTION_SIZE:
        raise SystemExit(f'Policy map has {len(entries)} entries, expected {PROJECT_ACTION_SIZE}.')
    unmapped = sum(1 for entry in entries if entry < 0)
    print(f'{PROJECT_ACTION_SIZE - unmapped} actions mapped; {unmapped} never observed legal and left unmapped.')
    return torch.tensor([entry if entry >= 0 else LC0_POLICY_SIZE for entry in entries], dtype=torch.int64)


def wdl_is_probability(backbone: nn.Module, wdl_output_index: int) -> bool:
    """Lc0 exports differ in whether the value head is softmaxed; decide it by measurement."""
    with torch.inference_mode():
        probe = torch.zeros((2, LC0_INPUT_PLANES, 8, 8), dtype=torch.float32)
        probe[:, LC0_INPUT_PLANES - 1] = 1.0
        wdl = backbone(probe)[wdl_output_index]
    sums = wdl.sum(dim=1)
    within_unit_range = bool(torch.all(wdl >= 0.0) and torch.all(wdl <= 1.0))
    sums_to_one = bool(torch.all((sums - 1.0).abs() < 1.0e-3))
    print(f'Value head probe: range ok {within_unit_range}, sums {sums.tolist()}')
    return within_unit_range and sums_to_one


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', type=Path, required=True, help='Lc0 network exported with `lc0 leela2onnx`.')
    parser.add_argument('--policy-map', type=Path, required=True, help='Output of build_lc0_policy_map.py.')
    parser.add_argument('--output', type=Path, required=True, help='TorchScript module to write.')
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    backbone = load_backbone(arguments.onnx)
    permutation = build_permutation(arguments.policy_map)
    policy_index, wdl_index = locate_outputs(backbone)
    model = Lc0TeacherModel(
        backbone,
        permutation,
        wdl_is_probability(backbone, wdl_index),
        policy_index,
        wdl_index,
    ).eval()

    with torch.inference_mode():
        sample = torch.zeros((FIXED_BATCH, LC0_INPUT_PLANES, 8, 8), dtype=torch.float32)
        sample[:, LC0_INPUT_PLANES - 1] = 1.0
        policy, wdl = model(sample)
    if policy.shape != (FIXED_BATCH, PROJECT_ACTION_SIZE) or wdl.shape != (FIXED_BATCH, 3):
        raise SystemExit(f'Wrapped model produced {policy.shape} and {wdl.shape}.')
    if not bool(torch.all((wdl.sum(dim=1) - 1.0).abs() < 1.0e-2)):
        raise SystemExit('Wrapped WDL output is not a probability distribution.')
    if not bool(torch.all(torch.isfinite(policy))):
        raise SystemExit('Wrapped policy output contains non-finite logits.')

    # Traced on the serving device so any constant the ONNX conversion creates inline lands there; the
    # search and the tools all serve on cuda:0.
    trace_device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(trace_device)
    scripted = torch.jit.trace(model, sample.to(trace_device))
    torch.jit.save(scripted, str(arguments.output))
    print(f'Wrote {arguments.output}')


if __name__ == '__main__':
    main()
