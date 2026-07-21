import type {
  CreateGameResponse,
  PlayTurnRequest,
  PlayTurnResponse,
} from "./contracts.ts";

export class ApiError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = "ApiError";
  }
}

export function normalizeApiBaseUrl(value: string | undefined): string {
  return (value ?? "").trim().replace(/\/+$/, "");
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (response.ok) {
    return (await response.json()) as T;
  }

  let detail = `Request failed (${response.status})`;
  try {
    const body = (await response.json()) as { detail?: string };
    if (body.detail) detail = body.detail;
  } catch {
    // The status still gives the user a useful error when the body is not JSON.
  }
  throw new ApiError(detail);
}

export class ChessApi {
  public constructor(private readonly baseUrl: string) {}

  public async createGame(
    startingFen: string,
    movesUci: readonly string[],
  ): Promise<CreateGameResponse> {
    const response = await fetch(`${this.baseUrl}/api/games`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ starting_fen: startingFen, moves_uci: movesUci }),
    });
    return parseResponse<CreateGameResponse>(response);
  }

  public async playTurn(
    gameToken: string,
    request: PlayTurnRequest,
  ): Promise<PlayTurnResponse> {
    const response = await fetch(
      `${this.baseUrl}/api/games/${encodeURIComponent(gameToken)}/turns`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      },
    );
    return parseResponse<PlayTurnResponse>(response);
  }

  public async endGame(gameToken: string): Promise<void> {
    const response = await fetch(
      `${this.baseUrl}/api/games/${encodeURIComponent(gameToken)}`,
      { method: "DELETE" },
    );
    if (!response.ok) await parseResponse<never>(response);
  }
}
