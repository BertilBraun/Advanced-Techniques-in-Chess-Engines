import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, ChessApi, normalizeApiBaseUrl } from "./api.ts";

afterEach(() => vi.unstubAllGlobals());

describe("normalizeApiBaseUrl", () => {
  it.each([
    [undefined, ""],
    ["", ""],
    [" https://api.example.test/// ", "https://api.example.test"],
  ])("normalizes %s", (input, expected) => {
    expect(normalizeApiBaseUrl(input)).toBe(expected);
  });
});

describe("ChessApi", () => {
  it("sends complete history with a human move", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          state: {
            starting_fen: "fen",
            moves_uci: ["e2e4", "e7e5"],
            fen: "next",
            side_to_move: "white",
            game_over: false,
            result: null,
          },
          engine_move_uci: "e7e5",
          analysis: null,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const api = new ChessApi("https://api.example.test");
    await api.playTurn("game-token", {
      starting_fen: "fen",
      moves_uci: ["e2e4"],
      human_move_uci: "g1f3",
      analysis: { mode: "mcts", time_limit_seconds: 12 },
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual({
      starting_fen: "fen",
      moves_uci: ["e2e4"],
      human_move_uci: "g1f3",
      analysis: { mode: "mcts", time_limit_seconds: 12 },
    });
  });

  it("surfaces FastAPI error details", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Illegal move." }), {
          status: 422,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );
    await expect(new ChessApi("").createGame("fen", [])).rejects.toEqual(
      new ApiError("Illegal move."),
    );
  });
});
