import { Chess } from "chess.js";
import { describe, expect, it } from "vitest";

import {
  STARTING_FEN,
  gameOutcomeMessage,
  historyRows,
  matchingLegalUci,
  percent,
  reconstructGame,
  signedValue,
} from "./game.ts";

describe("game helpers", () => {
  it("reconstructs complete UCI history and produces SAN rows", () => {
    const game = reconstructGame(STARTING_FEN, ["e2e4", "e7e5", "g1f3"]);
    expect(game.fen()).toContain(" b ");
    expect(historyRows(game)).toEqual([
      { number: 1, white: "e4", black: "e5" },
      { number: 2, white: "Nf3", black: null },
    ]);
  });

  it("accepts only a locally legal board move", () => {
    const game = new Chess();
    expect(matchingLegalUci(game, "e2", "e4")).toBe("e2e4");
    expect(matchingLegalUci(game, "e2", "e5")).toBeNull();
  });

  it("formats optional analysis values", () => {
    expect(percent(0.125)).toBe("12.5%");
    expect(signedValue(0.25)).toBe("+0.250");
    expect(signedValue(null)).toBe("N/A");
  });

  it("describes decisive game results", () => {
    const checkmate = reconstructGame(STARTING_FEN, ["f2f3", "e7e5", "g2g4", "d8h4"]);
    expect(gameOutcomeMessage(checkmate, "0-1")).toEqual({
      title: "Black wins",
      detail: "Black won by checkmate.",
    });
    expect(gameOutcomeMessage(new Chess(), "1-0")).toEqual({
      title: "White wins",
      detail: "White won the game.",
    });
  });

  it("describes the reason for a draw", () => {
    const stalemate = new Chess("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    expect(gameOutcomeMessage(stalemate, "1/2-1/2")).toEqual({
      title: "It's a draw",
      detail: "Draw by stalemate.",
    });
  });
});
