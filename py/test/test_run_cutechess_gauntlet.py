from pathlib import Path

from tools.run_cutechess_gauntlet import (
    GauntletConfiguration,
    command_for_opponent,
    prepare_openings,
)


def test_command_uses_paired_openings_and_fixed_move_time() -> None:
    configuration = GauntletConfiguration(
        cutechess=Path('cutechess-cli'),
        stockfish=Path('stockfish'),
        model=Path('best.jit.pt'),
        openings=Path('openings.epd'),
        output_directory=Path('results'),
        games_per_opponent=24,
        seconds_per_move=10,
        stockfish_threads=8,
        stockfish_hash_mib=1024,
        stockfish_elos=(1800,),
    )

    command = command_for_opponent(configuration, 1800)

    assert '-repeat' in command
    assert 'st=10' in command
    assert 'option.UCI_LimitStrength=true' in command
    assert 'option.UCI_Elo=1800' in command
    assert 'option.Threads=8' in command
    assert 'option.Hash=1024' in command
    assert 'policy=round' in command
    assert command[command.index('-games') + 1] == '2'
    assert command[command.index('-rounds') + 1] == '12'
    assert 'cmd=stockfish' in command
    assert all(',cmd=' not in argument for argument in command)


def test_prepare_openings_converts_tsv_and_selects_one_per_game_pair(
    tmp_path: Path,
) -> None:
    opening_suite = tmp_path / 'openings.tsv'
    opening_suite.write_text(
        '# provenance\nfirst\tfen one\nsecond\tfen two\nthird\tfen three\n',
        encoding='utf-8',
    )
    output_directory = tmp_path / 'results'
    output_directory.mkdir()
    configuration = GauntletConfiguration(
        cutechess=Path('cutechess-cli'),
        stockfish=Path('stockfish'),
        model=Path('best.jit.pt'),
        openings=opening_suite,
        output_directory=output_directory,
        games_per_opponent=4,
        seconds_per_move=5,
        stockfish_threads=1,
        stockfish_hash_mib=16,
        stockfish_elos=(2200,),
    )

    prepare_openings(configuration)

    assert configuration.prepared_openings.read_text(encoding='utf-8') == ('fen one\nfen two\n')
    command = command_for_opponent(configuration, 2200)
    assert f'file={configuration.prepared_openings}' in command
