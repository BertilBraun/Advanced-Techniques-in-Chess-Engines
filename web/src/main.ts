import { Chess } from "chess.js";
import {
  Chessboard,
  COLOR,
  INPUT_EVENT_TYPE,
  type MoveInputEvent,
} from "cm-chessboard/src/Chessboard.js";
import "cm-chessboard/assets/chessboard.css";
import piecesUrl from "cm-chessboard/assets/pieces/staunty.svg?url";

import { ChessApi, normalizeApiBaseUrl } from "./api.ts";
import type {
  AnalysisMode,
  AnalysisResult,
  GameState,
  PlayerSide,
} from "./contracts.ts";
import {
  STARTING_FEN,
  historyRows,
  matchingLegalUci,
  percent,
  reconstructGame,
  signedValue,
} from "./game.ts";
import "./style.css";

type UiPhase = "setup" | "cold" | "thinking" | "playing" | "game-over" | "error";

const root = document.querySelector<HTMLDivElement>("#app");
if (!root) throw new Error("App container is missing.");

root.innerHTML = `
  <header class="site-header">
    <a class="brand" href="./" aria-label="Neural Chess home">
      <span class="brand-mark" aria-hidden="true">N</span>
      <span>Neural Chess</span>
    </a>
    <span class="model-badge">AlphaZero model · live</span>
  </header>
  <main>
    <section class="hero" aria-labelledby="page-title">
      <p class="eyebrow">PLAY THE NETWORK</p>
      <h1 id="page-title">Your move.<br><em>Its calculation.</em></h1>
      <p class="intro">Play directly against the trained model. Choose a quick policy response or give MCTS time to search.</p>
    </section>

    <section class="play-layout">
      <div class="board-column">
        <div id="status-card" class="status-card" role="status" aria-live="polite">
          <span id="status-pulse" class="status-pulse"></span>
          <div><strong id="status-title">Configure a game</strong><span id="status-detail">Choose your side and search mode.</span></div>
        </div>
        <div id="board" class="board" aria-label="Interactive chessboard"></div>
        <div class="board-footer">
          <span id="turn-label">Ready when you are</span>
          <button id="new-game-inline" class="text-button" type="button" hidden>New game</button>
        </div>
      </div>

      <aside class="control-column">
        <section class="panel setup-panel" aria-labelledby="setup-title">
          <div class="panel-heading"><span>01</span><h2 id="setup-title">Game setup</h2></div>
          <fieldset>
            <legend>Your side</legend>
            <div class="segmented">
              <input id="side-white" name="side" type="radio" value="white" checked><label for="side-white"><span class="piece-dot light"></span>White</label>
              <input id="side-black" name="side" type="radio" value="black"><label for="side-black"><span class="piece-dot dark"></span>Black</label>
            </div>
          </fieldset>
          <fieldset>
            <legend>Engine mode</legend>
            <div class="mode-grid">
              <label class="mode-card"><input name="mode" type="radio" value="policy" checked><span><strong>Policy</strong><small>Instant network choice</small></span></label>
              <label class="mode-card"><input name="mode" type="radio" value="mcts"><span><strong>MCTS</strong><small>Tree search</small></span></label>
            </div>
          </fieldset>
          <div id="time-control" class="time-control" hidden>
            <label for="time-range">Thinking time <output id="time-output" for="time-range">5s</output></label>
            <input id="time-range" type="range" min="1" max="30" value="5">
            <div class="range-labels"><span>1 sec</span><span>30 sec</span></div>
          </div>
          <button id="start-game" class="primary-button" type="button">Start game <span aria-hidden="true">→</span></button>
        </section>

        <section class="panel analysis-panel" aria-labelledby="analysis-title">
          <div class="panel-heading"><span>02</span><h2 id="analysis-title">Engine analysis</h2></div>
          <div id="analysis-empty" class="empty-state">Analysis appears after the model makes a move.</div>
          <div id="analysis-content" hidden>
            <p class="perspective-note">W / D / L from the side-to-move at the search root</p>
            <div class="wdl" aria-label="Root outcome prediction">
              <div><span>Win</span><strong id="wdl-win">N/A</strong></div>
              <div><span>Draw</span><strong id="wdl-draw">N/A</strong></div>
              <div><span>Loss</span><strong id="wdl-loss">N/A</strong></div>
            </div>
            <dl class="metrics">
              <div><dt>Searches</dt><dd id="metric-searches">—</dd></div>
              <div><dt>Max depth</dt><dd id="metric-depth">—</dd></div>
              <div><dt>Elapsed</dt><dd id="metric-elapsed">—</dd></div>
              <div><dt>Root value</dt><dd id="metric-value">—</dd></div>
            </dl>
            <div class="pv"><span>Principal variation</span><code id="pv-value">N/A</code></div>
            <div class="candidate-heading"><h3>Candidate moves</h3><span>ranked</span></div>
            <div class="table-wrap">
              <table>
                <thead><tr><th>Move</th><th>Prior</th><th>Visits</th><th>Share</th><th>Value</th></tr></thead>
                <tbody id="candidate-body"></tbody>
              </table>
            </div>
          </div>
        </section>

        <section class="panel history-panel" aria-labelledby="history-title">
          <div class="panel-heading"><span>03</span><h2 id="history-title">Move history</h2></div>
          <ol id="move-history" class="move-history"><li class="history-empty">No moves yet.</li></ol>
        </section>
      </aside>
    </section>
  </main>
  <footer><span>Neural network chess</span><span>API-authoritative play · UCI history synchronized each turn</span></footer>
`;

function requiredElement<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing element: ${selector}`);
  return element;
}

const api = new ChessApi(normalizeApiBaseUrl(import.meta.env.VITE_API_BASE_URL));
const board = new Chessboard(requiredElement("#board"), {
  position: STARTING_FEN,
  orientation: COLOR.white,
  assetsUrl: "./",
  style: {
    cssClass: "chess-club",
    showCoordinates: true,
    pieces: { file: piecesUrl },
    animationDuration: 220,
  },
});

let game = new Chess(STARTING_FEN);
let gameToken: string | null = null;
let authoritativeState: GameState | null = null;
let playerSide: PlayerSide = "white";
let phase: UiPhase = "setup";

const startButton = requiredElement<HTMLButtonElement>("#start-game");
const newGameButton = requiredElement<HTMLButtonElement>("#new-game-inline");
const timeControl = requiredElement<HTMLDivElement>("#time-control");
const timeRange = requiredElement<HTMLInputElement>("#time-range");
const timeOutput = requiredElement<HTMLOutputElement>("#time-output");

function selectedValue<T extends string>(name: string): T {
  return requiredElement<HTMLInputElement>(`input[name="${name}"]:checked`).value as T;
}

function setPhase(nextPhase: UiPhase, detail?: string): void {
  phase = nextPhase;
  const card = requiredElement("#status-card");
  card.dataset.phase = nextPhase;
  const labels: Record<UiPhase, readonly [string, string]> = {
    setup: ["Configure a game", "Choose your side and search mode."],
    cold: ["Waking the engine", "A cold start can take a moment. Your game will begin automatically."],
    thinking: ["Engine is thinking", "Searching candidate lines. Please keep this tab open."],
    playing: ["Your move", "Drag a piece or click its destination."],
    "game-over": ["Game over", detail ?? authoritativeState?.result ?? "The game has ended."],
    error: ["Something went wrong", detail ?? "Please start a new game and try again."],
  };
  requiredElement("#status-title").textContent = labels[nextPhase][0];
  requiredElement("#status-detail").textContent = labels[nextPhase][1];
  startButton.disabled = nextPhase === "cold" || nextPhase === "thinking";
  newGameButton.hidden = nextPhase === "setup";
  if (nextPhase !== "playing") board.disableMoveInput();
}

function analysisOptions(): { mode: AnalysisMode; time_limit_seconds: number } {
  return {
    mode: selectedValue<AnalysisMode>("mode"),
    time_limit_seconds: Number(timeRange.value),
  };
}

function renderHistory(): void {
  const history = requiredElement<HTMLOListElement>("#move-history");
  const rows = historyRows(game);
  if (rows.length === 0) {
    history.innerHTML = '<li class="history-empty">No moves yet.</li>';
    return;
  }
  history.innerHTML = rows
    .map(
      (row) => `<li><span>${row.number}.</span><strong>${row.white}</strong><strong>${row.black ?? ""}</strong></li>`,
    )
    .join("");
  history.lastElementChild?.scrollIntoView({ block: "nearest" });
}

function renderAnalysis(analysis: AnalysisResult | null): void {
  requiredElement<HTMLDivElement>("#analysis-empty").hidden = analysis !== null;
  requiredElement<HTMLDivElement>("#analysis-content").hidden = analysis === null;
  if (!analysis) return;

  const outcome = analysis.outcome_prediction;
  requiredElement("#wdl-win").textContent = outcome ? percent(outcome.win) : "N/A";
  requiredElement("#wdl-draw").textContent = outcome ? percent(outcome.draw) : "N/A";
  requiredElement("#wdl-loss").textContent = outcome ? percent(outcome.loss) : "N/A";
  requiredElement("#metric-searches").textContent = analysis.metrics.searches.toLocaleString();
  requiredElement("#metric-depth").textContent = String(analysis.metrics.maximum_depth);
  requiredElement("#metric-elapsed").textContent = `${analysis.metrics.elapsed_milliseconds.toLocaleString()} ms`;
  requiredElement("#metric-value").textContent = signedValue(analysis.root_value);
  requiredElement("#pv-value").textContent = analysis.principal_variation?.join(" ") || "N/A";
  requiredElement<HTMLTableSectionElement>("#candidate-body").innerHTML = analysis.candidates
    .map(
      (candidate, index) => `
        <tr${index === 0 ? ' class="chosen"' : ""}>
          <td><span class="rank">${index + 1}</span><code>${candidate.move_uci}</code></td>
          <td>${percent(candidate.policy_prior)}</td>
          <td>${candidate.visits.toLocaleString()}</td>
          <td>${percent(candidate.visit_share)}</td>
          <td>${signedValue(candidate.mean_search_value)}</td>
        </tr>`,
    )
    .join("");
}

async function acceptState(state: GameState, animate: boolean): Promise<void> {
  authoritativeState = state;
  game = reconstructGame(state.starting_fen, state.moves_uci);
  await board.setPosition(state.fen, animate);
  renderHistory();
  requiredElement("#turn-label").textContent = state.game_over
    ? state.result ?? "Game over"
    : `${state.side_to_move === "white" ? "White" : "Black"} to move`;
  if (state.game_over) {
    setPhase("game-over", state.result ?? "The game has ended.");
  }
}

function enablePlayerInput(): void {
  if (!authoritativeState || authoritativeState.game_over) return;
  setPhase("playing");
  board.enableMoveInput(handleBoardInput, playerSide === "white" ? COLOR.white : COLOR.black);
}

async function requestTurn(humanMoveUci: string | null): Promise<void> {
  if (!gameToken || !authoritativeState) return;
  setPhase("thinking");
  try {
    const response = await api.playTurn(gameToken, {
      starting_fen: authoritativeState.starting_fen,
      moves_uci: authoritativeState.moves_uci,
      human_move_uci: humanMoveUci,
      analysis: analysisOptions(),
    });
    await acceptState(response.state, true);
    renderAnalysis(response.analysis);
    if (!response.state.game_over) enablePlayerInput();
  } catch (error) {
    await board.setPosition(authoritativeState.fen, true);
    setPhase("error", error instanceof Error ? error.message : "Unexpected API error.");
  }
}

function handleBoardInput(event: MoveInputEvent): boolean | void {
  if (phase !== "playing") return false;
  if (event.type === INPUT_EVENT_TYPE.moveInputStarted) return true;
  if (event.type === INPUT_EVENT_TYPE.validateMoveInput) {
    if (!event.squareTo) return false;
    return matchingLegalUci(game, event.squareFrom, event.squareTo) !== null;
  }
  if (event.type === INPUT_EVENT_TYPE.moveInputFinished && event.legalMove && event.squareTo) {
    const moveUci = matchingLegalUci(game, event.squareFrom, event.squareTo);
    if (moveUci) void requestTurn(moveUci);
  }
}

async function startGame(): Promise<void> {
  const previousToken = gameToken;
  gameToken = null;
  authoritativeState = null;
  if (previousToken) void api.endGame(previousToken).catch(() => undefined);
  playerSide = selectedValue<PlayerSide>("side");
  await board.setOrientation(playerSide === "white" ? COLOR.white : COLOR.black);
  await board.setPosition(STARTING_FEN, false);
  game = new Chess(STARTING_FEN);
  renderHistory();
  renderAnalysis(null);
  setPhase("cold");
  try {
    const response = await api.createGame(STARTING_FEN, []);
    gameToken = response.game_token;
    await acceptState(response.state, false);
    if (playerSide === "black" && !response.state.game_over) {
      await requestTurn(null);
    } else if (!response.state.game_over) {
      enablePlayerInput();
    }
  } catch (error) {
    setPhase("error", error instanceof Error ? error.message : "Unexpected API error.");
  }
}

document.querySelectorAll<HTMLInputElement>('input[name="mode"]').forEach((input) => {
  input.addEventListener("change", () => {
    timeControl.hidden = selectedValue<AnalysisMode>("mode") !== "mcts";
  });
});
timeRange.addEventListener("input", () => {
  timeOutput.value = `${timeRange.value}s`;
});
startButton.addEventListener("click", () => void startGame());
newGameButton.addEventListener("click", () => {
  setPhase("setup");
  requiredElement(".setup-panel").scrollIntoView({ behavior: "smooth", block: "start" });
});
window.addEventListener("pagehide", () => {
  if (gameToken) void api.endGame(gameToken);
});
