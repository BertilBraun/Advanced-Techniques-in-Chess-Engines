from __future__ import annotations

import sys
from pathlib import Path

import pytest
from tools.run_v34_final_evaluations import Arguments as FinalArguments
from tools.run_v34_final_evaluations import child_commands as final_commands
from tools.run_v34_replay_distillation import Arguments as DistillationArguments
from tools.run_v34_replay_distillation import training_arms, training_commands
from tools.run_v34_stockfish_ladders import Arguments as LadderArguments
from tools.run_v34_stockfish_ladders import child_commands as ladder_commands
from tools.v34_orchestration import ChildCommand, run_child_commands


def _ladder_arguments(tmp_path: Path) -> LadderArguments:
    return LadderArguments(
        experiment=tmp_path / 'experiment.yaml',
        run_directory=tmp_path / 'run',
        checkpoint_generation=123,
        opening_manifest=tmp_path / 'openings.json',
        stockfish_executable=tmp_path / 'stockfish',
        output_root=tmp_path / 'ladders',
        ten_thousand_devices=(0, 1, 2),
        eighty_thousand_devices=(3, 4, 5, 6, 7),
        opening_selection_seed=20260815,
        match_random_seed=20260816,
        dry_run=True,
    )


def test_ladder_commands_pin_the_protocol_and_disjoint_devices(tmp_path: Path) -> None:
    commands = ladder_commands(_ladder_arguments(tmp_path))

    assert len(commands) == 2
    assert commands[0].command[
        commands[0].command.index('--devices') + 1 : commands[0].command.index('--model-searches')
    ] == (
        '0',
        '1',
        '2',
    )
    assert commands[1].command[
        commands[1].command.index('--devices') + 1 : commands[1].command.index('--model-searches')
    ] == (
        '3',
        '4',
        '5',
        '6',
        '7',
    )
    assert ('--model-searches', '10000', '--parallel-searches', '4') == commands[0].command[
        commands[0].command.index('--model-searches') : commands[0].command.index('--inference-workers')
    ]
    assert ('--model-searches', '80000', '--parallel-searches', '8') == commands[1].command[
        commands[1].command.index('--model-searches') : commands[1].command.index('--inference-workers')
    ]


def test_final_commands_include_true_policy_only_and_three_search_budgets(tmp_path: Path) -> None:
    arguments = FinalArguments(
        experiment=tmp_path / 'experiment.yaml',
        run_directory=tmp_path / 'run',
        checkpoint_generation=123,
        opening_manifest=tmp_path / 'openings.json',
        stockfish_executable=tmp_path / 'stockfish',
        output_root=tmp_path / 'final',
        devices=tuple(range(8)),
        policy_only_stockfish_nodes=1_000,
        shallow_stockfish_nodes=2_000,
        deep_stockfish_nodes=50_000,
        very_deep_stockfish_nodes=100_000,
        match_random_seed=20260816,
        dry_run=True,
    )

    commands = final_commands(arguments)

    assert '--model-policy-only' in commands[0].command
    assert '--model-searches' not in commands[0].command
    assert tuple(command.command[command.command.index('--model-searches') + 1] for command in commands[1:]) == (
        '64',
        '10000',
        '80000',
    )
    assert all(command.command[command.command.index('--inference-batch-size') + 1] == '64' for command in commands)


def test_child_supervisor_waits_for_all_children_and_retains_logs(tmp_path: Path) -> None:
    successful_log = tmp_path / 'successful.log'
    failed_log = tmp_path / 'failed.log'
    outcomes = run_child_commands(
        (
            ChildCommand('successful', (sys.executable, '-c', "print('complete')"), successful_log),
            ChildCommand('failed', (sys.executable, '-c', "print('failure'); raise SystemExit(7)"), failed_log),
        ),
        dry_run=False,
        poll_interval_seconds=0.01,
    )

    assert {outcome.name: outcome.return_code for outcome in outcomes} == {'failed': 7, 'successful': 0}
    assert successful_log.read_text(encoding='utf-8').strip() == 'complete'
    assert failed_log.read_text(encoding='utf-8').strip() == 'failure'


def test_dry_run_starts_no_children_and_writes_no_logs(tmp_path: Path) -> None:
    log_path = tmp_path / 'unused.log'

    outcomes = run_child_commands(
        (ChildCommand('unused', (sys.executable, '-c', "raise SystemExit('should not run')"), log_path),),
        dry_run=True,
    )

    assert outcomes == ()
    assert not log_path.exists()


def _distillation_arguments(tmp_path: Path) -> DistillationArguments:
    return DistillationArguments(
        teacher_run_state=tmp_path / 'teacher',
        teacher_generation=123,
        replay_store=tmp_path / 'replay.bin',
        experiment=tmp_path / 'experiment.yaml',
        opening_manifest=tmp_path / 'openings.json',
        output_root=tmp_path / 'distillation',
        devices=tuple(range(8)),
        seeds=(41, 43),
        student_generation=0,
        steps=100_000,
        searches_per_move=10_000,
        parallel_searches=4,
        throughput_device=0,
        dry_run=True,
    )


def test_distillation_sweep_assigns_two_seeds_per_architecture_across_all_gpus(tmp_path: Path) -> None:
    arguments = _distillation_arguments(tmp_path)

    arms = training_arms(arguments)
    commands = training_commands(arguments, '0' * 64)

    assert tuple(arm.architecture.name for arm in arms) == (
        '4x80',
        '4x80',
        '5x72',
        '5x72',
        '6x64',
        '6x64',
        '8x56',
        '8x56',
    )
    assert tuple(arm.device_id for arm in arms) == tuple(range(8))
    assert all('--replay-store' in command.command for command in commands)
    assert all('--policy-head-kind' in command.command for command in commands)
    assert all(command.command[command.command.index('--policy-key-size') + 1] == '64' for command in commands)


def test_distillation_sweep_skips_completed_checkpoint_and_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arguments = _distillation_arguments(tmp_path)
    completed = training_arms(arguments)[0]

    def completed_first(run_state: Path, log_path: Path, generation: int) -> bool:
        return run_state.name == completed.name

    monkeypatch.setattr(
        'tools.run_v34_replay_distillation._arm_is_complete',
        completed_first,
    )

    commands = training_commands(arguments, '0' * 64)

    assert len(commands) == 7
    assert completed.name not in {command.name for command in commands}
