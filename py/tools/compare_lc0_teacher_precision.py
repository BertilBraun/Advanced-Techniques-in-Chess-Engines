"""Compares a reduced-precision Lc0 teacher with the float32 one it was traced from, and times both.

The float32 teacher is the one the fidelity gate verified against real Lc0. A float16 trace is accepted as a
data-generation teacher only if it chooses the same top move and near-identical priors on positions games
actually reach, so it is scored against the float32 teacher on replayed match positions, not on synthetic ones.

Requires the native extension, so it runs on the node.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from src.distillation.lc0_teacher import teacher_input_dtype
from src.games.chess.contract import CHESS_STATE_CONTRACT, decode_lc0_planes
from tools.measure_teacher_agreement import TEACHER_BATCH, legal_softmax, match_positions


def evaluate(model: torch.jit.ScriptModule, planes: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    dtype = teacher_input_dtype(model)
    policies: list[np.ndarray] = []
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(planes), TEACHER_BATCH):
            chunk = torch.from_numpy(planes[start : start + TEACHER_BATCH]).to(device=device, dtype=dtype)
            policy, wdl = model(chunk)
            policies.append(policy.float().cpu().numpy())
            values.append(wdl.float().cpu().numpy())
    return np.concatenate(policies), np.concatenate(values)


def throughput(model: torch.jit.ScriptModule, device: torch.device, seconds: float) -> float:
    batch = torch.zeros((TEACHER_BATCH, 112, 8, 8), device=device, dtype=teacher_input_dtype(model))
    batch[:, 111] = 1
    with torch.inference_mode():
        for _ in range(5):
            model(batch)
        torch.cuda.synchronize(device)
        calls = 0
        started = time.perf_counter()
        while time.perf_counter() - started < seconds:
            model(batch)
            calls += 1
        torch.cuda.synchronize(device)
    return calls * TEACHER_BATCH / (time.perf_counter() - started)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='The gated float32 teacher.')
    parser.add_argument('--candidate', type=Path, required=True, help='A reduced-precision trace.')
    parser.add_argument('--match-result', action='append', type=Path, required=True)
    parser.add_argument('--positions', type=int, default=5000)
    parser.add_argument('--timing-seconds', type=float, default=10.0)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    device = torch.device(arguments.device)
    reference = torch.jit.load(str(arguments.reference), map_location=device).eval()
    candidate = torch.jit.load(str(arguments.candidate), map_location=device).eval()
    print(f'Reference dtype {teacher_input_dtype(reference)}, candidate dtype {teacher_input_dtype(candidate)}')

    replayed = match_positions(arguments.match_result, candidate_moves_only=False)[: arguments.positions]
    positions = [position for position, _ in replayed]
    planes = decode_lc0_planes(tuple(position.lc0_packed_encoding() for position in positions)).astype(np.float32)
    reference_policy, reference_wdl = evaluate(reference, planes, device)
    candidate_policy, candidate_wdl = evaluate(candidate, planes, device)

    agreements = 0
    worst_prior = 0.0
    prior_differences: list[float] = []
    for row, position in enumerate(positions):
        legal = np.asarray(CHESS_STATE_CONTRACT.legal_action_ids(position), dtype=np.int64)
        ours = legal_softmax(candidate_policy[row], legal)
        theirs = legal_softmax(reference_policy[row], legal)
        agreements += int(np.argmax(ours) == np.argmax(theirs))
        difference = float(np.abs(ours - theirs).max())
        prior_differences.append(difference)
        worst_prior = max(worst_prior, difference)
    worst_wdl = float(np.abs(candidate_wdl - reference_wdl).max())

    print(f'Positions compared: {len(positions)}')
    print(f'Top-move agreement with float32: {agreements / len(positions):.4f}')
    print(f'Prior difference: mean {np.mean(prior_differences):.5f}, worst {worst_prior:.5f}')
    print(f'Worst WDL difference: {worst_wdl:.5f}')
    reference_rate = throughput(reference, device, arguments.timing_seconds)
    candidate_rate = throughput(candidate, device, arguments.timing_seconds)
    print(
        f'Throughput at batch {TEACHER_BATCH}: reference {reference_rate:.0f}/s, candidate {candidate_rate:.0f}/s, '
        f'ratio {candidate_rate / reference_rate:.2f}'
    )


if __name__ == '__main__':
    main()
