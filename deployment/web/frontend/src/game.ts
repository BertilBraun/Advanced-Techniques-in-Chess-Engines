import { Chess, type Move } from "chess.js";

export const STARTING_FEN =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export interface HistoryEntry {
  readonly number: number;
  readonly white: string;
  readonly black: string | null;
}

export interface GameOutcomeMessage {
  readonly title: string;
  readonly detail: string;
}

export function matchingLegalUci(
  game: Chess,
  from: string,
  to: string,
): string | null {
  const candidates = game
    .moves({ square: from as Move["from"], verbose: true })
    .filter((candidate) => candidate.to === to);
  const move = candidates.find((candidate) => candidate.promotion === "q") ?? candidates[0];
  if (!move) return null;
  return `${from}${to}${move.promotion ?? ""}`;
}

export function reconstructGame(
  startingFen: string,
  movesUci: readonly string[],
): Chess {
  const game = new Chess(startingFen);
  for (const moveUci of movesUci) {
    const move = game.move({
      from: moveUci.slice(0, 2),
      to: moveUci.slice(2, 4),
      promotion: moveUci.slice(4, 5) || undefined,
    });
    if (!move) throw new Error(`Server returned an invalid move: ${moveUci}`);
  }
  return game;
}

export function historyRows(game: Chess): readonly HistoryEntry[] {
  const moves = game.history();
  const rows: HistoryEntry[] = [];
  for (let index = 0; index < moves.length; index += 2) {
    rows.push({
      number: index / 2 + 1,
      white: moves[index],
      black: moves[index + 1] ?? null,
    });
  }
  return rows;
}

export function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function signedValue(value: number | null): string {
  return value === null ? "N/A" : `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

export function gameOutcomeMessage(
  game: Chess,
  result: string | null,
): GameOutcomeMessage {
  if (result === "1-0") {
    return {
      title: "White wins",
      detail: game.isCheckmate() ? "White won by checkmate." : "White won the game.",
    };
  }
  if (result === "0-1") {
    return {
      title: "Black wins",
      detail: game.isCheckmate() ? "Black won by checkmate." : "Black won the game.",
    };
  }
  if (result === "1/2-1/2") {
    if (game.isStalemate()) return { title: "It's a draw", detail: "Draw by stalemate." };
    if (game.isThreefoldRepetition()) {
      return { title: "It's a draw", detail: "Draw by threefold repetition." };
    }
    if (game.isInsufficientMaterial()) {
      return { title: "It's a draw", detail: "Draw by insufficient material." };
    }
    if (game.isDrawByFiftyMoves()) {
      return { title: "It's a draw", detail: "Draw by the fifty-move rule." };
    }
    return { title: "It's a draw", detail: "The game ended in a draw." };
  }
  return { title: "Game over", detail: result ?? "The game has ended." };
}
