from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.training.model_cost import measure_model_cost
from src.training.network import Network
from tools.attention_viability_cells import ALL_CELLS, cell_by_name
from tools.distill_train_student import student_architecture

EVALUATION_LINE = re.compile(
    r'step (?P<step>\d+)/(?P<total>\d+) lr (?P<learning_rate>[\d.e+-]+) '
    r'(?P<steps_per_second>[\d.]+) steps/s (?P<samples_per_second>[\d.]+) samples/s '
    r'peak (?P<peak_mebibytes>[\d.]+) MiB \| '
    r'train policy (?P<train_policy>[\d.]+) wdl (?P<train_wdl>[\d.]+) total (?P<train_total>[\d.]+) \| '
    r'held-out policy (?P<held_out_policy>[\d.]+) wdl (?P<held_out_wdl>[\d.]+) total (?P<held_out_total>[\d.]+) \| '
    r'headline policy gap above floor (?P<policy_gap>[\d.-]+)'
)
FLOOR_LINE = re.compile(r'Held-out loss floor: policy (?P<policy>[\d.]+), wdl (?P<wdl>[\d.]+)')


@dataclass(frozen=True)
class Evaluation:
    step: int
    steps_per_second: float
    samples_per_second: float
    peak_mebibytes: float
    train_policy: float
    held_out_policy: float
    held_out_wdl: float
    policy_gap_above_floor: float


@dataclass(frozen=True)
class CellReport:
    cell: str
    question: str
    completed_steps: int
    requested_steps: int
    policy_floor: float
    final_policy_gap_above_floor: float
    final_held_out_wdl: float
    train_held_out_separation: float
    median_samples_per_second: float
    peak_mebibytes: float
    total_parameters: int
    trunk_parameters: int
    policy_head_parameters: int
    value_head_parameters: int
    multiply_accumulates_total: int
    multiply_accumulates_trunk: int
    multiply_accumulates_policy_head: int
    evaluations: tuple[Evaluation, ...]


def parse_log(path: Path) -> tuple[float, tuple[Evaluation, ...], int]:
    policy_floor = 0.0
    requested_steps = 0
    evaluations: list[Evaluation] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        floor = FLOOR_LINE.search(line)
        if floor is not None:
            policy_floor = float(floor.group('policy'))
        match = EVALUATION_LINE.search(line)
        if match is None:
            continue
        requested_steps = int(match.group('total'))
        evaluations.append(
            Evaluation(
                step=int(match.group('step')),
                steps_per_second=float(match.group('steps_per_second')),
                samples_per_second=float(match.group('samples_per_second')),
                peak_mebibytes=float(match.group('peak_mebibytes')),
                train_policy=float(match.group('train_policy')),
                held_out_policy=float(match.group('held_out_policy')),
                held_out_wdl=float(match.group('held_out_wdl')),
                policy_gap_above_floor=float(match.group('policy_gap')),
            )
        )
    return policy_floor, tuple(evaluations), requested_steps


def median(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def build_report(cell_name: str, log_path: Path, truncate_at_step: int) -> CellReport:
    cell = cell_by_name(cell_name)
    policy_floor, evaluations, requested_steps = parse_log(log_path)
    if truncate_at_step:
        evaluations = tuple(evaluation for evaluation in evaluations if evaluation.step <= truncate_at_step)
    if not evaluations:
        raise ValueError(f'{log_path} carries no evaluation lines.')
    final = evaluations[-1]
    model = Network(student_architecture(cell.arguments), torch.device('cpu'), CHESS_NETWORK_DIMENSIONS)
    cost = measure_model_cost(model)
    # The first window includes warm-up and allocator growth, so it is excluded from the rate.
    steady = tuple(evaluation.samples_per_second for evaluation in evaluations[1:]) or (final.samples_per_second,)
    return CellReport(
        cell=cell.name,
        question=cell.question,
        completed_steps=final.step,
        requested_steps=requested_steps,
        policy_floor=policy_floor,
        final_policy_gap_above_floor=final.policy_gap_above_floor,
        final_held_out_wdl=final.held_out_wdl,
        train_held_out_separation=final.held_out_policy - final.train_policy,
        median_samples_per_second=median(steady),
        peak_mebibytes=max(evaluation.peak_mebibytes for evaluation in evaluations),
        total_parameters=cost.parameters.total,
        trunk_parameters=cost.parameters.trunk,
        policy_head_parameters=cost.parameters.policy_head,
        value_head_parameters=cost.parameters.value_head,
        multiply_accumulates_total=cost.multiply_accumulates_per_position.total,
        multiply_accumulates_trunk=cost.multiply_accumulates_per_position.trunk,
        multiply_accumulates_policy_head=cost.multiply_accumulates_per_position.policy_head,
        evaluations=evaluations,
    )


def format_table(reports: tuple[CellReport, ...]) -> str:
    header = (
        '| cell | steps | CE gap above floor | held-out WDL CE | total params | trunk | policy head | '
        'trunk MAC | total MAC | samples/s (rtx3060) | peak MiB |'
    )
    separator = '| --- ' * 11 + '|'
    rows = [
        f'| {report.cell} | {report.completed_steps:,} | {report.final_policy_gap_above_floor:.4f} | '
        f'{report.final_held_out_wdl:.4f} | {report.total_parameters:,} | {report.trunk_parameters:,} | '
        f'{report.policy_head_parameters:,} | {report.multiply_accumulates_trunk:,} | '
        f'{report.multiply_accumulates_total:,} | {report.median_samples_per_second:.0f} | '
        f'{report.peak_mebibytes:.0f} |'
        for report in reports
    ]
    return '\n'.join((header, separator, *rows))


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect the attention-viability cell logs into one table.')
    parser.add_argument('--log-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--truncate-at-step', default=0, type=int, help='Compare every cell at a common step.')
    namespace = parser.parse_args()

    reports: list[CellReport] = []
    for cell in ALL_CELLS:
        log_path = namespace.log_root / f'{cell.name}.log'
        if not log_path.is_file():
            print(f'{log_path} is absent; skipping {cell.name}.')
            continue
        reports.append(build_report(cell.name, log_path, namespace.truncate_at_step))

    namespace.output.parent.mkdir(parents=True, exist_ok=True)
    namespace.output.write_text(
        json.dumps([asdict(report) for report in reports], indent=2),
        encoding='utf-8',
    )
    print(format_table(tuple(reports)))
    print(f'\nWrote {namespace.output}.')


if __name__ == '__main__':
    main()
