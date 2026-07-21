from __future__ import annotations

from dataclasses import dataclass

import chess
from fastapi.testclient import TestClient

from web_play.api import create_app
from web_play.contracts import (
    AnalysisLimit,
    AnalysisResult,
    CandidateMove,
    SearchMetrics,
)
from web_play.service import GameService


@dataclass
class FakeGame:
    board: chess.Board

    @property
    def fen(self) -> str:
        return self.board.fen()

    @property
    def is_game_over(self) -> bool:
        return self.board.is_game_over(claim_draw=True)

    @property
    def result(self) -> str | None:
        return self.board.result(claim_draw=True) if self.is_game_over else None

    def apply_move(self, move_uci: str) -> None:
        try:
            self.board.push(self.board.parse_uci(move_uci))
        except ValueError as error:
            raise ValueError(f"Illegal UCI move {move_uci!r}.") from error

    def analyze(self, limit: AnalysisLimit) -> AnalysisResult:
        del limit
        move = min(move.uci() for move in self.board.legal_moves)
        return AnalysisResult(
            chosen_move_uci=move,
            root_value=0.25,
            outcome_prediction=None,
            candidates=(
                CandidateMove(
                    move_uci=move,
                    policy_prior=0.4,
                    visits=8,
                    visit_share=1.0,
                    mean_search_value=0.2,
                ),
            ),
            metrics=SearchMetrics(searches=8, maximum_depth=3, elapsed_milliseconds=12),
            principal_variation=(move,),
        )


class FakeEngine:
    def __init__(self) -> None:
        self.creations = 0

    def new_game(self, starting_fen: str, moves_uci: tuple[str, ...]) -> FakeGame:
        self.creations += 1
        board = chess.Board(starting_fen)
        for move_uci in moves_uci:
            board.push(board.parse_uci(move_uci))
        return FakeGame(board)


def test_turn_uses_complete_history_and_returns_authoritative_moves() -> None:
    engine = FakeEngine()
    client = TestClient(create_app(GameService(engine), ["https://chess.example"]))
    created = client.post(
        "/api/games", json={"starting_fen": chess.STARTING_FEN, "moves_uci": []}
    )
    token = created.json()["game_token"]

    response = client.post(
        f"/api/games/{token}/turns",
        json={
            "starting_fen": chess.STARTING_FEN,
            "moves_uci": [],
            "human_move_uci": "e2e4",
            "analysis": {"mode": "mcts", "time_limit_seconds": 1},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"]["moves_uci"][0] == "e2e4"
    assert payload["state"]["moves_uci"][1] == payload["engine_move_uci"]
    assert payload["analysis"]["metrics"]["searches"] == 8
    assert engine.creations == 1


def test_unknown_uuid_recovers_from_complete_history() -> None:
    engine = FakeEngine()
    client = TestClient(create_app(GameService(engine), ["https://chess.example"]))

    response = client.post(
        "/api/games/9c36b28e-2bd7-4979-bbbc-2eb20fb76ad8/turns",
        json={
            "starting_fen": chess.STARTING_FEN,
            "moves_uci": ["e2e4", "e7e5"],
            "human_move_uci": "g1f3",
            "analysis": {"mode": "policy", "time_limit_seconds": 1},
        },
    )

    assert response.status_code == 200
    assert response.json()["state"]["moves_uci"][:3] == ["e2e4", "e7e5", "g1f3"]
    assert engine.creations == 1


def test_illegal_move_and_invalid_time_are_rejected() -> None:
    client = TestClient(
        create_app(GameService(FakeEngine()), ["https://chess.example"])
    )
    token = client.post(
        "/api/games", json={"starting_fen": chess.STARTING_FEN, "moves_uci": []}
    ).json()["game_token"]
    request = {
        "starting_fen": chess.STARTING_FEN,
        "moves_uci": [],
        "human_move_uci": "e2e5",
        "analysis": {"mode": "mcts", "time_limit_seconds": 31},
    }
    assert client.post(f"/api/games/{token}/turns", json=request).status_code == 422
    request["analysis"]["time_limit_seconds"] = 1
    assert client.post(f"/api/games/{token}/turns", json=request).status_code == 422


def test_cors_is_limited_to_configured_origin() -> None:
    client = TestClient(
        create_app(GameService(FakeEngine()), ["https://chess.example"])
    )
    allowed = client.options(
        "/api/games",
        headers={
            "Origin": "https://chess.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    denied = client.options(
        "/api/games",
        headers={
            "Origin": "https://wrong.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert allowed.headers["access-control-allow-origin"] == "https://chess.example"
    assert denied.status_code == 400
