// Shared contract between the TypeScript engines, the API client and the UI.
// Shapes mirror the FastAPI responses (backend/app/main.py) so both data sources are interchangeable.

export type Sport = 'football' | 'mma'

export interface Competition { id: number; name: string; sport: Sport; country: string | null; season: string; avg_goals_per_match: number }

export interface Team {
  id: number; name: string; short_name: string | null; competition_id: number; elo: number; matches_played: number
  xg_for_avg: number; xg_against_avg: number; goals_for_avg: number; goals_against_avg: number; possession_avg: number
  shots_per_match: number; shots_on_target_per_match: number; ppda: number; defensive_rating: number
  clean_sheet_rate: number; form: string
}

export interface Fighter {
  id: number; name: string; nickname: string | null; competition_id: number | null; weight_class: string | null
  stance: string | null; age: number | null; height_cm: number | null; reach_cm: number | null
  wins: number; losses: number; draws: number; ko_wins: number; sub_wins: number; dec_wins: number
  slpm: number; sapm: number; striking_accuracy: number; striking_defense: number; td_avg: number
  td_accuracy: number; td_defense: number; sub_avg: number; cardio: number; elo: number
}

export interface SportEvent {
  id: number; sport: Sport; competition_id: number; event_date: string; status: 'scheduled' | 'finished'
  home_team_id: number | null; away_team_id: number | null; fighter_a_id: number | null; fighter_b_id: number | null
  home_goals: number | null; away_goals: number | null; home_xg: number | null; away_xg: number | null
  home_possession: number | null; winner_fighter_id: number | null
  method: 'ko_tko' | 'submission' | 'decision' | 'draw' | 'no_contest' | null
  end_round: number | null; scheduled_rounds: number
  odds_home: number | null; odds_draw: number | null; odds_away: number | null
  odds_fighter_a: number | null; odds_fighter_b: number | null
}

export interface Dataset { competitions: Competition[]; teams: Team[]; fighters: Fighter[]; events: SportEvent[] }

/** Live what-if adjustments, applied identically by the Python and TypeScript engines. */
export interface FootballAdjustments {
  home_advantage: number // multiplier on the home edge, 1 = model default, range 0.8..1.3
  home_form: number // -0.2..0.2, attack multiplier offset for the home side
  away_form: number
  home_absences: number // 0..5 key players missing
  away_absences: number
}
export interface FightAdjustments { form_a: number; form_b: number } // -0.2..0.2, Elo shift = form * 250

export interface FootballSimulationRequest {
  home_team_id: number; away_team_id: number; n_simulations: number; seed: number | null
  rho: number; lam3: number; max_goals: number
  odds: { home: number; draw: number; away: number } | null
  bankroll: number; kelly_multiplier: number; adjustments: FootballAdjustments
}
export interface FightSimulationRequest {
  fighter_a_id: number; fighter_b_id: number; rounds: 3 | 5; n_simulations: number; seed: number | null
  odds: { fighter_a: number; fighter_b: number } | null
  bankroll: number; kelly_multiplier: number; adjustments: FightAdjustments
}

export interface ScoreCell { score: string; home_goals: number; away_goals: number; probability: number }
export interface FootballMarkets {
  home_win: number; draw: number; away_win: number; btts_yes: number; btts_no: number
  over_under: Record<string, { over: number; under: number }>
  expected_home_goals: number; expected_away_goals: number; correct_scores: ScoreCell[]
  double_chance: { '1X': number; X2: number; '12': number }; asian_handicap_0: { home: number; away: number }
}
export interface FootballPrediction {
  home: string; away: string; lambda_home: number; lambda_away: number; rho: number; lam3: number
  elo_home: number; elo_away: number; elo_win_expectancy: number; matrix: number[][]; markets: FootballMarkets
}
export interface FootballSimulationSummary {
  n_simulations: number; elapsed_ms: number; home_win: number; draw: number; away_win: number; btts_yes: number
  over_under: Record<string, { over: number; under: number }>
  mean_home_goals: number; mean_away_goals: number; std_home_goals: number; std_away_goals: number
  confidence_intervals: { home_win: [number, number]; draw: [number, number]; away_win: [number, number] }
  score_distribution: { score: string; frequency: number }[]
  goal_histogram: { home: number[]; away: number[] }
}

export type MethodProbs = { ko_tko: number; submission: number; decision: number }
export interface MMAPrediction {
  fighter_a: string; fighter_b: string; style_a: string; style_b: string; win_prob_a: number; win_prob_b: number
  method_probs: { a: MethodProbs; b: MethodProbs }; expected_rounds: number
  finish_round_probs: Record<string, number>; factors: Record<string, number>
}
export interface FightSimulationSummary {
  n_simulations: number; elapsed_ms: number; win_prob_a: number; win_prob_b: number
  confidence_intervals: { a: [number, number] }
  methods: { a: MethodProbs; b: MethodProbs }; round_distribution: Record<string, number>; finish_rate: number
}

export interface ValueBet {
  selection: string; model_prob: number; decimal_odds: number; implied_prob: number; fair_prob: number
  edge: number; expected_value: number; kelly_fraction: number; stake: number; is_value: boolean
}

export interface FootballSimulationResult {
  prediction: FootballPrediction; simulation: FootballSimulationSummary
  value_bets: ValueBet[] | null; market_margin: number | null
}
export interface FightSimulationResult {
  prediction: MMAPrediction; simulation: FightSimulationSummary
  value_bets: ValueBet[] | null; market_margin: number | null
}

export interface CalibrationBin { bin_low: number; bin_high: number; mean_predicted: number | null; observed_frequency: number | null; count: number }
export interface BacktestRequest { warmup: number; bankroll: number; kelly_multiplier: number; min_ev: number }
export interface BacktestResult {
  n_events: number; n_bets: number; brier_score: number; log_loss: number; accuracy: number
  calibration: CalibrationBin[]; roi: number; yield_pct: number; profit: number; final_bankroll: number
  max_drawdown: number; sharpe: number; bankroll_curve: number[]
  bets: { index: number; selection: string; model_prob: number; odds: number; stake: number; profit: number; bankroll: number }[]
  n_matches: number; warmup: number
  predictions: { date: string; home: string; away: string; home_goals: number; away_goals: number; p_home: number; p_draw: number; p_away: number; outcome: 'H' | 'D' | 'A' }[]
}

/** One interface, two implementations: the FastAPI backend, or the in-browser engines on the bundled dataset. */
export interface QuantService {
  mode: 'api' | 'local'
  loadDataset(): Promise<Dataset>
  simulateFootball(req: FootballSimulationRequest): Promise<FootballSimulationResult>
  simulateFight(req: FightSimulationRequest): Promise<FightSimulationResult>
  backtestFootball(req: BacktestRequest): Promise<BacktestResult>
}
