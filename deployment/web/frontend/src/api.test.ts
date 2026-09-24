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
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
      .mockResolvedValueOnce(
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
      analysis: { type: "timed_mcts", seconds: 12 },
    });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0]).toEqual([
      "https://api.example.test/api/ready",
      { method: "GET" },
    ]);
    const [, init] = fetchMock.mock.calls[1] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual({
      starting_fen: "fen",
      moves_uci: ["e2e4"],
      human_move_uci: "g1f3",
      analysis: { type: "timed_mcts", seconds: 12 },
    });
  });

  it("surfaces FastAPI error details", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(new Response(null, { status: 204 }))
        .mockResolvedValueOnce(
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

  it("explains network failures from a busy or starting engine", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    await expect(new ChessApi("").createGame("fen", [])).rejects.toEqual(
      new ApiError(
        "The chess engine did not respond. It may be starting up or analyzing another game. Wait a moment, then try again.",
      ),
    );
  });

  it("explains transient gateway responses without an API detail", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 503 })));

    await expect(new ChessApi("").createGame("fen", [])).rejects.toEqual(
      new ApiError(
        "The chess engine did not respond. It may be starting up or analyzing another game. Wait a moment, then try again.",
      ),
    );
  });

  it("explains when Modal disables the workspace after its GPU budget is exhausted", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response("modal-http: workspace ac-example is disabled\n", { status: 404 }),
      ),
    );

    await expect(new ChessApi("").createGame("fen", [])).rejects.toEqual(
      new ApiError(
        "The monthly GPU allowance has been used up. Sorry—the chess engine will be available again when the €30 monthly compute credits reset.",
      ),
    );
  });

  it("does not describe an ordinary missing endpoint as a budget error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("Not found", { status: 404 })),
    );

    await expect(new ChessApi("").createGame("fen", [])).rejects.toEqual(
      new ApiError("Request failed (404)"),
    );
  });
});
