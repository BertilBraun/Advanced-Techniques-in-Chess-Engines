export type AnalysisMode = "policy" | "mcts";
export type PlayerSide = "white" | "black";

export interface AnalysisOptions {
  readonly mode: AnalysisMode;
  readonly time_limit_seconds: number;
}

export interface GameState {
  readonly starting_fen: string;
  readonly moves_uci: readonly string[];
  readonly fen: string;
  readonly side_to_move: PlayerSide;
  readonly game_over: boolean;
  readonly result: string | null;
}

export interface OutcomePrediction {
  readonly win: number;
  readonly draw: number;
  readonly loss: number;
  readonly perspective: "side_to_move";
}

export interface CandidateMove {
  readonly move_uci: string;
  readonly policy_prior: number;
  readonly visits: number;
  readonly visit_share: number;
  readonly mean_search_value: number | null;
}

export interface SearchMetrics {
  readonly searches: number;
  readonly maximum_depth: number;
  readonly elapsed_milliseconds: number;
}

export interface AnalysisResult {
  readonly chosen_move_uci: string;
  readonly root_value: number;
  readonly outcome_prediction: OutcomePrediction | null;
  readonly candidates: readonly CandidateMove[];
  readonly metrics: SearchMetrics;
  readonly principal_variation: readonly string[] | null;
}

export interface CreateGameResponse {
  readonly game_token: string;
  readonly state: GameState;
}

export interface PlayTurnResponse {
  readonly state: GameState;
  readonly engine_move_uci: string | null;
  readonly analysis: AnalysisResult | null;
}

export interface PlayTurnRequest {
  readonly starting_fen: string;
  readonly moves_uci: readonly string[];
  readonly human_move_uci: string | null;
  readonly analysis: AnalysisOptions;
}
