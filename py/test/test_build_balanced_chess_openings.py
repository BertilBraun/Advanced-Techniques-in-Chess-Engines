from pathlib import Path

from tools.build_balanced_chess_openings import _extract_candidates


def test_candidate_extraction_deduplicates_transpositions_and_filters_rating(tmp_path: Path) -> None:
    pgn = tmp_path / 'games.pgn'
    pgn.write_text(
        """[Event "First"]
[Result "1/2-1/2"]
[WhiteElo "2600"]
[BlackElo "2550"]
[ECO "A05"]
[Opening "Reti"]

1. Nf3 d5 2. g3 Nf6 3. Bg2 g6 4. O-O Bg7 1/2-1/2

[Event "Transposition"]
[Result "1-0"]
[WhiteElo "2520"]
[BlackElo "2510"]
[ECO "A05"]
[Opening "Reti"]

1. g3 d5 2. Nf3 Nf6 3. Bg2 g6 4. O-O Bg7 1-0

[Event "Below threshold"]
[Result "0-1"]
[WhiteElo "2499"]
[BlackElo "2600"]
[ECO "A05"]
[Opening "Reti"]

1. Nf3 d5 2. g3 Nf6 3. Bg2 g6 4. O-O Bg7 0-1
""",
        encoding='utf-8',
    )

    candidates, games_read, eligible_games = _extract_candidates(pgn, 2_500, 8)

    assert games_read == 3
    assert eligible_games == 2
    assert len(candidates) == 1
    assert candidates[0].frequency == 2
    assert candidates[0].eco_code == 'A05'
    assert candidates[0].final_fen.split()[:4] == [
        'rnbqk2r/ppp1ppbp/5np1/3p4/8/5NP1/PPPPPPBP/RNBQ1RK1',
        'w',
        'kq',
        '-',
    ]
