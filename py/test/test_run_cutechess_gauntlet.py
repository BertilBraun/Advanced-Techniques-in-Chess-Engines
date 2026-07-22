from pathlib import Path

from tools.run_cutechess_gauntlet import GauntletConfiguration, command_for_opponent


def test_command_uses_paired_openings_and_fixed_move_time() -> None:
    configuration = GauntletConfiguration(
        cutechess=Path("cutechess-cli"),
        stockfish=Path("stockfish"),
        model=Path("best.jit.pt"),
        openings=Path("openings.epd"),
        output_directory=Path("results"),
        games_per_opponent=24,
        seconds_per_move=10,
        stockfish_elos=(1800,),
    )

    command = command_for_opponent(configuration, 1800)

    assert "-repeat" in command
    assert "st=10" in command
    assert "option.UCI_LimitStrength=true" in command
    assert "option.UCI_Elo=1800" in command
    assert "policy=round" in command
