import type { AnalysisMode, GameState, PlayerSide } from "./contracts.ts";
import { reconstructGame } from "./game.ts";

const STORAGE_KEY = "neural-chess.active-game.v1";
const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export interface BrowserStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface StoredGame {
  readonly version: 1;
  readonly game_token: string;
  readonly state: GameState;
  readonly player_side: PlayerSide;
  readonly analysis_mode: AnalysisMode;
  readonly time_limit_seconds: number;
}

export function loadStoredGame(storage: BrowserStorage): StoredGame | null {
  let serialized: string | null;
  try {
    serialized = storage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
  if (serialized === null) return null;

  try {
    const value: unknown = JSON.parse(serialized);
    return parseStoredGame(value);
  } catch {
    return null;
  }
}

export function saveStoredGame(storage: BrowserStorage, game: StoredGame): boolean {
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(game));
    return true;
  } catch {
    return false;
  }
}

export function clearStoredGame(storage: BrowserStorage): boolean {
  try {
    storage.removeItem(STORAGE_KEY);
    return true;
  } catch {
    return false;
  }
}

function parseStoredGame(value: unknown): StoredGame | null {
  if (!isRecord(value) || value.version !== 1) return null;
  if (typeof value.game_token !== "string" || !UUID_PATTERN.test(value.game_token)) {
    return null;
  }
  if (value.player_side !== "white" && value.player_side !== "black") return null;
  if (value.analysis_mode !== "policy" && value.analysis_mode !== "mcts") return null;
  if (
    typeof value.time_limit_seconds !== "number" ||
    !Number.isInteger(value.time_limit_seconds) ||
    value.time_limit_seconds < 1 ||
    value.time_limit_seconds > 30
  ) {
    return null;
  }

  const state = parseGameState(value.state);
  if (state === null) return null;
  return {
    version: 1,
    game_token: value.game_token,
    state,
    player_side: value.player_side,
    analysis_mode: value.analysis_mode,
    time_limit_seconds: value.time_limit_seconds,
  };
}

function parseGameState(value: unknown): GameState | null {
  if (!isRecord(value)) return null;
  if (
    typeof value.starting_fen !== "string" ||
    !isStringArray(value.moves_uci) ||
    typeof value.fen !== "string" ||
    (value.side_to_move !== "white" && value.side_to_move !== "black") ||
    typeof value.game_over !== "boolean" ||
    (value.result !== null && typeof value.result !== "string")
  ) {
    return null;
  }

  const reconstructed = reconstructGame(value.starting_fen, value.moves_uci);
  if (reconstructed.fen() !== value.fen) return null;
  return {
    starting_fen: value.starting_fen,
    moves_uci: value.moves_uci,
    fen: value.fen,
    side_to_move: value.side_to_move,
    game_over: value.game_over,
    result: value.result,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}
