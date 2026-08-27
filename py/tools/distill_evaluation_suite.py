from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

STUDENT_NAMES = ('6x64', '6x96', '14x64', '8x96')
TEACHER_GENERATION = 322
STUDENT_GENERATION = 322


@dataclass(frozen=True)
class Tier:
    name: str
    searches_per_move: int


@dataclass(frozen=True)
class Matchup:
    student_name: str
    tier: Tier
    mode: str

    @property
    def label(self) -> str:
        return f'{self.student_name}-{self.tier.name}-{self.mode}'


@dataclass(frozen=True)
class Arguments:
    teacher_run_state: Path
    students_root: Path
    openings_manifest: Path
    experiment_config: Path
    output_directory: Path
    log_path: Path
    python_executable: Path
    working_directory: Path
    tiers: tuple[Tier, ...]
    modes: tuple[str, ...]
    student_names: tuple[str, ...]
    opening_pair_count: int
    parallel_searches: int
    exploration_constant: float
    maximum_game_plies: int
    bootstrap_samples: int
    device_id: int
    random_seed: int


def _log(log_path: Path, message: str) -> None:
    stamped = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}'
    print(stamped, flush=True)
    with log_path.open('a', encoding='utf-8') as handle:
        handle.write(stamped + '\n')


def _matchups(arguments: Arguments) -> tuple[Matchup, ...]:
    return tuple(
        Matchup(student_name=student_name, tier=tier, mode=mode)
        for tier in arguments.tiers
        for mode in arguments.modes
        for student_name in arguments.student_names
    )


def _output_path(arguments: Arguments, matchup: Matchup) -> Path:
    return arguments.output_directory / f'distill-match-{matchup.label}.json'


def _command(arguments: Arguments, matchup: Matchup, output_path: Path) -> tuple[str, ...]:
    return (
        str(arguments.python_executable),
        '-m',
        'tools.distill_match',
        '--teacher-run-state',
        str(arguments.teacher_run_state),
        '--teacher-generation',
        str(TEACHER_GENERATION),
        '--student-run-state',
        str(arguments.students_root / matchup.student_name),
        '--student-generation',
        str(STUDENT_GENERATION),
        '--openings-manifest',
        str(arguments.openings_manifest),
        '--experiment-config',
        str(arguments.experiment_config),
        '--mode',
        matchup.mode,
        '--searches-per-move',
        str(matchup.tier.searches_per_move),
        '--parallel-searches',
        str(arguments.parallel_searches),
        '--exploration-constant',
        str(arguments.exploration_constant),
        '--opening-pair-count',
        str(arguments.opening_pair_count),
        '--maximum-game-plies',
        str(arguments.maximum_game_plies),
        '--bootstrap-samples',
        str(arguments.bootstrap_samples),
        '--device-id',
        str(arguments.device_id),
        '--random-seed',
        str(arguments.random_seed),
        '--output',
        str(output_path),
    )


def run_suite(arguments: Arguments) -> int:
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    arguments.log_path.parent.mkdir(parents=True, exist_ok=True)
    matchups = _matchups(arguments)
    _log(arguments.log_path, f'suite start: {len(matchups)} matchups, {arguments.opening_pair_count * 2} games each')
    failed: list[str] = []
    for index, matchup in enumerate(matchups, start=1):
        output_path = _output_path(arguments, matchup)
        prefix = f'[{index}/{len(matchups)}] {matchup.label}'
        if output_path.exists():
            _log(arguments.log_path, f'{prefix}: skipped, {output_path} already exists')
            continue
        student_run_state = arguments.students_root / matchup.student_name
        if not student_run_state.is_dir():
            _log(arguments.log_path, f'{prefix}: FAILED, student run state missing: {student_run_state}')
            failed.append(matchup.label)
            continue
        _log(arguments.log_path, f'{prefix}: starting, teacher budget {matchup.tier.searches_per_move}')
        started_at = time.monotonic()
        completed = subprocess.run(
            _command(arguments, matchup, output_path),
            cwd=arguments.working_directory,
            check=False,
        )
        elapsed = time.monotonic() - started_at
        if completed.returncode != 0:
            _log(arguments.log_path, f'{prefix}: FAILED with exit {completed.returncode} after {elapsed:.0f} s')
            failed.append(matchup.label)
            continue
        _log(arguments.log_path, f'{prefix}: done in {elapsed:.0f} s -> {output_path}')
    if failed:
        _log(arguments.log_path, f'suite finished with {len(failed)} failed matchups: {", ".join(failed)}')
        return 1
    _log(arguments.log_path, 'suite finished: every matchup produced a result')
    return 0


def _parse_tier(value: str) -> Tier:
    name, _, searches = value.partition('=')
    if not name or not searches.isdigit() or int(searches) <= 0:
        raise ValueError(f'Tier must be NAME=SEARCHES with a positive search count, got: {value}')
    return Tier(name=name, searches_per_move=int(searches))


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description='Run the full teacher-versus-student distillation evaluation suite.')
    parser.add_argument('--teacher-run-state', default=Path('/workspace/distill/teacher-v8-g322'), type=Path)
    parser.add_argument('--students-root', default=Path('/workspace/distill/students'), type=Path)
    parser.add_argument(
        '--openings-manifest',
        default=Path('/workspace/evaluation-artifacts/chess/chess-elite-2025-11-balanced-4moves-200-v1-openings.json'),
        type=Path,
    )
    parser.add_argument(
        '--experiment-config',
        default=Path('configs/validation/distillation-probe-chess-single-gpu.yaml'),
        type=Path,
    )
    parser.add_argument('--output-directory', default=Path('/workspace/distill/evaluation'), type=Path)
    parser.add_argument('--log-path', default=Path('/workspace/distill/evaluation/suite.log'), type=Path)
    parser.add_argument('--python-executable', default=Path('/workspace/alphazero-engine-venv/bin/python'), type=Path)
    parser.add_argument('--working-directory', default=Path('/workspace/alphazero-engine/py'), type=Path)
    parser.add_argument('--tier', action='append', dest='tiers', default=None, type=_parse_tier)
    parser.add_argument('--mode', action='append', dest='modes', default=None, choices=('equal-nodes', 'equal-compute'))
    parser.add_argument('--student', action='append', dest='student_names', default=None, choices=STUDENT_NAMES)
    parser.add_argument('--opening-pair-count', default=100, type=int)
    parser.add_argument('--parallel-searches', default=1, type=int)
    parser.add_argument('--exploration-constant', default=1.0, type=float)
    parser.add_argument('--maximum-game-plies', default=300, type=int)
    parser.add_argument('--bootstrap-samples', default=10000, type=int)
    parser.add_argument('--device-id', default=0, type=int)
    parser.add_argument('--random-seed', default=0, type=int)
    namespace = parser.parse_args()
    if not namespace.tiers:
        raise ValueError('At least one --tier NAME=SEARCHES is required; derive it from a throughput-only measurement.')
    tier_names = tuple(tier.name for tier in namespace.tiers)
    if len(set(tier_names)) != len(tier_names):
        raise ValueError('Tier names must be unique.')
    return Arguments(
        teacher_run_state=namespace.teacher_run_state,
        students_root=namespace.students_root,
        openings_manifest=namespace.openings_manifest,
        experiment_config=namespace.experiment_config,
        output_directory=namespace.output_directory,
        log_path=namespace.log_path,
        python_executable=namespace.python_executable,
        working_directory=namespace.working_directory,
        tiers=tuple(namespace.tiers),
        modes=tuple(namespace.modes or ('equal-compute',)),
        student_names=tuple(namespace.student_names or STUDENT_NAMES),
        opening_pair_count=namespace.opening_pair_count,
        parallel_searches=namespace.parallel_searches,
        exploration_constant=namespace.exploration_constant,
        maximum_game_plies=namespace.maximum_game_plies,
        bootstrap_samples=namespace.bootstrap_samples,
        device_id=namespace.device_id,
        random_seed=namespace.random_seed,
    )


def main() -> None:
    sys.exit(run_suite(parse_arguments()))


if __name__ == '__main__':
    main()
