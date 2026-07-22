import { describe, expect, it } from "vitest";

import { STARTING_FEN, reconstructGame } from "./game.ts";
import {
  clearStoredGame,
  loadStoredGame,
  saveStoredGame,
  type BrowserStorage,
  type StoredGame,
} from "./persistence.ts";

class MemoryStorage implements BrowserStorage {
  private value: string | null = null;

  public getItem(_key: string): string | null {
    return this.value;
  }

  public setItem(_key: string, value: string): void {
    this.value = value;
  }

  public removeItem(_key: string): void {
    this.value = null;
  }
}

function storedGame(): StoredGame {
  const moves = ["e2e4", "e7e5"];
  const game = reconstructGame(STARTING_FEN, moves);
  return {
    version: 1,
    game_token: "11ec79cb-7ac1-4a0e-90db-c66cc077d658",
    state: {
      starting_fen: STARTING_FEN,
      moves_uci: moves,
      fen: game.fen(),
      side_to_move: "white",
      game_over: false,
      result: null,
    },
    player_side: "white",
    analysis_mode: "mcts",
    time_limit_seconds: 5,
  };
}

describe("game persistence", () => {
  it("round-trips an authoritative game and its controls", () => {
    const storage = new MemoryStorage();
    const saved = storedGame();

    expect(saveStoredGame(storage, saved)).toBe(true);
    expect(loadStoredGame(storage)).toEqual(saved);
  });

  it("rejects a stored state whose FEN disagrees with its UCI history", () => {
    const storage = new MemoryStorage();
    const saved = storedGame();
    storage.setItem(
      "ignored",
      JSON.stringify({ ...saved, state: { ...saved.state, fen: STARTING_FEN } }),
    );

    expect(loadStoredGame(storage)).toBeNull();
  });

  it("clears a saved game", () => {
    const storage = new MemoryStorage();
    saveStoredGame(storage, storedGame());

    expect(clearStoredGame(storage)).toBe(true);
    expect(loadStoredGame(storage)).toBeNull();
  });
});
