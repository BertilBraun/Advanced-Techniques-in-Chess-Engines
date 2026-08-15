from pathlib import Path

from tools.run_stockfish_ladder import LadderProbe, _score_bracket


def _probe(nodes: int, score: float) -> LadderProbe:
    return LadderProbe(
        stockfish_nodes=nodes,
        result_path=Path(f'{nodes}.json'),
        wins=0,
        draws=20,
        losses=0,
        score=score,
        score_confidence_low=max(0.0, score - 0.2),
        score_confidence_high=min(1.0, score + 0.2),
    )


def test_score_bracket_finds_adjacent_node_rungs() -> None:
    bracket = _score_bracket((_probe(5_000, 0.3), _probe(1_000, 0.7), _probe(2_000, 0.55)))

    assert bracket is not None
    assert bracket.lower_stockfish_nodes == 2_000
    assert bracket.upper_stockfish_nodes == 5_000


def test_score_bracket_is_absent_without_crossing() -> None:
    assert _score_bracket((_probe(1_000, 0.8), _probe(2_000, 0.7))) is None
