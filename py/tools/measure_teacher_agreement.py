"""Measures how closely project networks follow the Lc0 teacher on the positions that matter.

Held-out loss on the distillation dataset fell by half while playing strength did not move. This asks why, on
two position sets side by side:

- positions reached in the actual evaluation matches, replayed from their result files, where the teacher is
  queried on Lc0 planes built from the real game history; and
- held-out rows of the distillation dataset itself, whose stored teacher policy is used directly.

For each network it reports top-move agreement with the teacher, the probability it puts on the teacher's top
move, and the KL divergence from the teacher over legal moves, split by game phase. If the students agree with
the teacher on dataset rows but not on match positions, the gap is distribution shift; if they do not agree on
the teacher's top move even where they imitate it on average, the loss is not measuring what decides games.

Requires the native extension, so it runs on the node.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from src.distillation.dataset import open_dataset
from src.distillation.lc0_teacher import teacher_input_dtype
from src.evaluation.inference import decode_packed_inputs
from src.games.chess.contract import CHESS_STATE_CONTRACT, ChessPosition, decode_lc0_planes
from src.games.representation import PackedPlanePayload

TEACHER_BATCH = 64
PHASES: tuple[tuple[str, int, int], ...] = (('opening', 0, 30), ('middlegame', 30, 80), ('late', 80, 10_000))


@dataclass(frozen=True)
class EvaluatedPosition:
    phase: str
    legal_action_ids: np.ndarray
    teacher_policy: np.ndarray


@dataclass
class Agreement:
    positions: int = 0
    top_move_agreements: int = 0
    teacher_move_probability: float = 0.0
    kl_divergence: float = 0.0
    by_phase: dict[str, list[float]] = field(default_factory=dict)

    def add(self, phase: str, agrees: bool, teacher_move_probability: float, kl_divergence: float) -> None:
        self.positions += 1
        self.top_move_agreements += int(agrees)
        self.teacher_move_probability += teacher_move_probability
        self.kl_divergence += kl_divergence
        totals = self.by_phase.setdefault(phase, [0.0, 0.0, 0.0, 0.0])
        totals[0] += 1
        totals[1] += int(agrees)
        totals[2] += teacher_move_probability
        totals[3] += kl_divergence

    def summary(self) -> dict[str, object]:
        count = max(self.positions, 1)
        return {
            'positions': self.positions,
            'top_move_agreement': self.top_move_agreements / count,
            'teacher_move_probability': self.teacher_move_probability / count,
            'kl_divergence': self.kl_divergence / count,
            'by_phase': {
                phase: {
                    'positions': int(totals[0]),
                    'top_move_agreement': totals[1] / max(totals[0], 1),
                    'teacher_move_probability': totals[2] / max(totals[0], 1),
                    'kl_divergence': totals[3] / max(totals[0], 1),
                }
                for phase, totals in self.by_phase.items()
            },
        }


def phase_of(ply: int) -> str:
    return next(name for name, low, high in PHASES if low <= ply < high)


def legal_softmax(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    values = logits[legal].astype(np.float64)
    shifted = np.exp(values - values.max())
    return shifted / shifted.sum()


def match_positions(result_paths: list[Path], candidate_moves_only: bool) -> list[tuple[ChessPosition, int]]:
    positions: list[tuple[ChessPosition, int]] = []
    for path in result_paths:
        for game in json.loads(path.read_text(encoding='utf-8'))['games']:
            position = CHESS_STATE_CONTRACT.initial_position()
            for action_id in game['initial_action_ids']:
                position = CHESS_STATE_CONTRACT.child_position(position, action_id)
            opening_plies = len(game['initial_action_ids'])
            candidate_parity = 0 if game['candidate_player'] == 'first' else 1
            for index, action_id in enumerate(game['played_action_ids']):
                if not candidate_moves_only or index % 2 == candidate_parity:
                    positions.append((position, opening_plies + index))
                position = CHESS_STATE_CONTRACT.child_position(position, action_id)
    return positions


def teacher_policies(
    teacher: torch.jit.ScriptModule, positions: list[ChessPosition], device: torch.device
) -> list[np.ndarray]:
    policies: list[np.ndarray] = []
    for start in range(0, len(positions), TEACHER_BATCH):
        chunk = positions[start : start + TEACHER_BATCH]
        planes = decode_lc0_planes(tuple(position.lc0_packed_encoding() for position in chunk)).astype(np.float32)
        with torch.inference_mode():
            logits, _ = teacher(torch.from_numpy(planes).to(device=device, dtype=teacher_input_dtype(teacher)))
        values = logits.float().cpu().numpy()
        for row, position in enumerate(chunk):
            legal = np.asarray(CHESS_STATE_CONTRACT.legal_action_ids(position), dtype=np.int64)
            policies.append(legal_softmax(values[row], legal))
    return policies


def network_logits(model: torch.jit.ScriptModule, packed_states: list[bytes], device: torch.device) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, len(packed_states), 256):
        payloads = tuple(PackedPlanePayload(state) for state in packed_states[start : start + 256])
        planes = decode_packed_inputs(CHESS_STATE_CONTRACT, payloads).astype(np.float32)
        with torch.inference_mode():
            logits, _ = model(torch.from_numpy(planes).to(device))
        rows.append(logits.float().cpu().numpy())
    return np.concatenate(rows)


def score(
    name: str,
    model: torch.jit.ScriptModule,
    packed_states: list[bytes],
    evaluated: list[EvaluatedPosition],
    device: torch.device,
) -> Agreement:
    logits = network_logits(model, packed_states, device)
    agreement = Agreement()
    for row, position in enumerate(evaluated):
        ours = legal_softmax(logits[row], position.legal_action_ids)
        teacher = position.teacher_policy
        teacher_top = int(np.argmax(teacher))
        support = teacher > 0
        kl = float(np.sum(teacher[support] * (np.log(teacher[support]) - np.log(np.maximum(ours[support], 1e-12)))))
        agreement.add(position.phase, int(np.argmax(ours)) == teacher_top, float(ours[teacher_top]), kl)
    print(f'  {name}: {agreement.summary()["top_move_agreement"]:.3f} top-move agreement')
    return agreement


def dataset_positions(dataset_path: Path, rows: int, seed: int) -> tuple[list[bytes], list[EvaluatedPosition]]:
    records, manifest = open_dataset(dataset_path)
    held_out_start = len(records) - int(len(records) * 0.02)
    generator = np.random.default_rng(seed)
    chosen = np.sort(generator.choice(np.arange(held_out_start, len(records)), size=rows, replace=False))
    states: list[bytes] = []
    evaluated: list[EvaluatedPosition] = []
    for index in chosen:
        record = records[index]
        legal = np.asarray(record['legal_action_ids'][: record['legal_count']], dtype=np.int64)
        teacher = np.zeros(len(legal), dtype=np.float64)
        position_of = {int(action): slot for slot, action in enumerate(legal)}
        for action, probability in zip(
            record['policy_action_ids'][: record['policy_count']],
            record['policy_probabilities'][: record['policy_count']],
            strict=True,
        ):
            teacher[position_of[int(action)]] = float(probability)
        teacher /= teacher.sum()
        states.append(bytes(record['packed_state']))
        # The dataset keeps no ply, so its rows are reported as one phase.
        evaluated.append(EvaluatedPosition(phase='dataset', legal_action_ids=legal, teacher_policy=teacher))
    return states, evaluated


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-model', type=Path, required=True)
    parser.add_argument('--network', action='append', required=True, help='name=path to a project .jit.pt')
    parser.add_argument('--match-result', action='append', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--dataset-rows', type=int, default=20_000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=20260926)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    device = torch.device(arguments.device)
    teacher = torch.jit.load(str(arguments.teacher_model), map_location=device).eval()
    networks = {
        name: torch.jit.load(path, map_location=device).eval()
        for name, path in (entry.split('=', 1) for entry in arguments.network)
    }

    replayed = match_positions(arguments.match_result, candidate_moves_only=True)
    positions = [position for position, _ in replayed]
    print(f'Replayed {len(positions)} match positions where the evaluated model was to move.')
    teacher_rows = teacher_policies(teacher, positions, device)
    match_states = [bytes(CHESS_STATE_CONTRACT.encode_network_input(position).payload) for position in positions]
    match_evaluated = [
        EvaluatedPosition(
            phase=phase_of(ply),
            legal_action_ids=np.asarray(CHESS_STATE_CONTRACT.legal_action_ids(position), dtype=np.int64),
            teacher_policy=policy,
        )
        for (position, ply), policy in zip(replayed, teacher_rows, strict=True)
    ]
    dataset_states, dataset_evaluated = dataset_positions(arguments.dataset, arguments.dataset_rows, arguments.seed)
    print(f'Sampled {len(dataset_states)} held-out dataset rows.')

    report: dict[str, dict[str, object]] = {}
    print('Match positions:')
    for name, model in networks.items():
        report.setdefault(name, {})['match'] = score(name, model, match_states, match_evaluated, device).summary()
    print('Held-out dataset rows:')
    for name, model in networks.items():
        report[name]['dataset'] = score(name, model, dataset_states, dataset_evaluated, device).summary()

    arguments.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
