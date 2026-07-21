from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware

from web_play.contracts import (
    CreateGameRequest,
    CreateGameResponse,
    PlayTurnRequest,
    PlayTurnResponse,
)
from web_play.service import GameService


def create_app(service: GameService, allowed_origins: Sequence[str]) -> FastAPI:
    if not allowed_origins:
        raise ValueError("At least one explicit CORS origin is required.")
    app = FastAPI(title="Chess Model Web Play API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed_origins),
        allow_credentials=False,
        allow_methods=["POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @app.post(
        "/api/games",
        response_model=CreateGameResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_game(request: CreateGameRequest) -> CreateGameResponse:
        try:
            return service.create_game(request.starting_fen, request.moves_uci)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error

    @app.post("/api/games/{game_token}/turns", response_model=PlayTurnResponse)
    def play_turn(game_token: UUID, request: PlayTurnRequest) -> PlayTurnResponse:
        try:
            return service.play_turn(game_token, request)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error

    @app.delete("/api/games/{game_token}", status_code=status.HTTP_204_NO_CONTENT)
    def end_game(game_token: UUID) -> Response:
        service.end_game(game_token)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app
