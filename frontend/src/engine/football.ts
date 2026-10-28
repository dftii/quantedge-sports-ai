// Port of backend/app/engines/models_football.py: same formulas, constants, clipping and defaults.
// Score matrices have rows = home goals, columns = away goals.
// Not ported: DixonColesModel (MLE fit needs an optimiser; the browser uses the profile-based predictor only).
import type { FootballAdjustments, FootballMarkets, FootballPrediction, ScoreCell } from './types'

export const ELO_LOG_SCALE = 500 // Elo-implied log goal ratio per rating point
export const POSSESSION_COEF = 0.003 // log-lambda tilt per possession point of difference
export const EWMA_ALPHA = 0.15 // smoothing of the xG averages in the walk-forward backtest
export const DEFAULT_OU_LINES = [0.5, 1.5, 2.5, 3.5, 4.5]
const LAMBDA_MIN = 0.05
const LAMBDA_MAX = 6

export interface TeamProfile {
  name: string
  elo: number
  xg_for: number // avg xG created per match
  xg_against: number // avg xG conceded per match
  possession: number // avg possession %
  defensive_rating: number // > 1 = better than league-average defence
}

export function teamProfile(p: Partial<TeamProfile> & { name: string }): TeamProfile {
  return { elo: 1500, xg_for: 1.35, xg_against: 1.35, possession: 50, defensive_rating: 1, ...p }
}

/** ISO dates (YYYY-MM-DD) sort chronologically as strings. */
export interface MatchRecord {
  date: string
  home: string
  away: string
  home_goals: number
  away_goals: number
  home_xg?: number | null
  away_xg?: number | null
  odds_home?: number | null
  odds_draw?: number | null
  odds_away?: number | null
}

export interface WalkForwardRow {
  date: string; home: string; away: string; home_goals: number; away_goals: number
  p_home: number; p_draw: number; p_away: number; p_over_2_5: number; p_btts: number
  outcome: 'H' | 'D' | 'A'
  odds_home: number | null; odds_draw: number | null; odds_away: number | null
}

// --------------------------------------------------------------------------- distributions
function checkMaxGoals(maxGoals: number): void {
  if (!Number.isInteger(maxGoals) || maxGoals < 1) throw new Error(`max_goals must be an integer >= 1, got ${maxGoals}`)
}

const LOG_FACTORIAL = [0]
function logFactorial(k: number): number {
  for (let i = LOG_FACTORIAL.length; i <= k; i++) LOG_FACTORIAL.push(LOG_FACTORIAL[i - 1] + Math.log(i))
  return LOG_FACTORIAL[k]
}

/** Poisson pmf for 0..maxGoals in log space (the tail beyond maxGoals is dropped). */
export function poissonPmfVector(lam: number, maxGoals: number): number[] {
  checkMaxGoals(maxGoals)
  if (lam < 0) throw new Error(`lam must be >= 0, got ${lam}`)
  return Array.from({ length: maxGoals + 1 }, (_, k) =>
    lam === 0 ? (k === 0 ? 1 : 0) : Math.exp(k * Math.log(lam) - lam - logFactorial(k)))
}

function normalised(m: number[][]): number[][] {
  let total = 0
  for (const row of m) for (const v of row) total += v
  return m.map((row) => row.map((v) => v / total))
}

/** Karlis-Ntzoufras bivariate Poisson grid: P(x, y) = sum_k p3[k] p1[x-k] p2[y-k], renormalised to sum 1. */
export function bivariatePoissonMatrix(lam1: number, lam2: number, lam3: number, maxGoals = 10): number[][] {
  checkMaxGoals(maxGoals)
  if (lam1 <= 0 || lam2 <= 0 || lam3 < 0) throw new Error(`need lam1, lam2 > 0 and lam3 >= 0, got ${lam1}, ${lam2}, ${lam3}`)
  const p1 = poissonPmfVector(lam1, maxGoals)
  const p2 = poissonPmfVector(lam2, maxGoals)
  const p3 = poissonPmfVector(lam3, maxGoals)
  return normalised(Array.from({ length: maxGoals + 1 }, (_, x) =>
    Array.from({ length: maxGoals + 1 }, (_, y) => {
      let s = 0
      for (let k = 0; k <= Math.min(x, y); k++) s += p3[k] * p1[x - k] * p2[y - k]
      return s
    })))
}

/** Range of rho for which every Dixon-Coles tau is >= 0. */
export function rhoBounds(lam: number, mu: number): [number, number] {
  return [Math.max(-1 / lam, -1 / mu), Math.min(1, 1 / (lam * mu))]
}

export function dixonColesTau(x: number, y: number, lam: number, mu: number, rho: number): number {
  if (x === 0 && y === 0) return 1 - lam * mu * rho
  if (x === 0 && y === 1) return 1 + lam * rho
  if (x === 1 && y === 0) return 1 + mu * rho
  if (x === 1 && y === 1) return 1 - rho
  return 1
}

/** Score grid with marginal means (lam, mu), shared covariance term lam3 and DC correction rho (clipped to its feasible range). */
export function dixonColesMatrix(lam: number, mu: number, rho: number, lam3 = 0, maxGoals = 10): number[][] {
  checkMaxGoals(maxGoals)
  if (lam <= 0 || mu <= 0) throw new Error(`lam and mu must be > 0, got ${lam}, ${mu}`)
  if (!(lam3 >= 0 && lam3 < Math.min(lam, mu))) throw new Error(`lam3 must satisfy 0 <= lam3 < min(lam, mu), got ${lam3}`)
  const [lo, hi] = rhoBounds(lam, mu)
  const r = Math.min(Math.max(rho, lo), hi)
  const base = bivariatePoissonMatrix(lam - lam3, mu - lam3, lam3, maxGoals)
  return normalised(base.map((row, x) => row.map((v, y) => v * dixonColesTau(x, y, lam, mu, r))))
}

// --------------------------------------------------------------------------- markets
/** Betting-market probabilities from a (home x away) score grid (normalised internally). */
export function marketProbabilities(matrix: number[][], ouLines: readonly number[] = DEFAULT_OU_LINES, topN = 10): FootballMarkets {
  const n = matrix.length
  let sum = 0
  for (const row of matrix) for (const v of row) sum += v
  if (n === 0 || matrix.some((r) => r.length !== n) || !(sum > 0)) throw new Error('matrix must be a square, non-empty score grid')
  const m = matrix.map((row) => row.map((v) => v / sum))

  let homeWin = 0, draw = 0, awayWin = 0, btts = 0, eh = 0, ea = 0
  const overs = ouLines.map(() => 0)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const p = m[i][j]
      if (i > j) homeWin += p
      else if (i === j) draw += p
      else awayWin += p
      if (i > 0 && j > 0) btts += p
      eh += p * i
      ea += p * j
      ouLines.forEach((line, l) => { if (i + j > line) overs[l] += p })
    }
  }

  const over_under: FootballMarkets['over_under'] = {}
  ouLines.forEach((line, l) => { over_under[String(line)] = { over: overs[l], under: 1 - overs[l] } })

  const flat = m.flat()
  const best = flat.map((_, k) => k).sort((a, b) => flat[b] - flat[a]).slice(0, Math.min(topN, flat.length))
  const correct_scores: ScoreCell[] = best.map((k) => {
    const i = Math.floor(k / n), j = k % n
    return { score: `${i}-${j}`, home_goals: i, away_goals: j, probability: flat[k] }
  })

  const decisive = homeWin + awayWin
  return {
    home_win: homeWin, draw, away_win: awayWin, btts_yes: btts, btts_no: 1 - btts, over_under,
    expected_home_goals: eh, expected_away_goals: ea, correct_scores,
    double_chance: { '1X': homeWin + draw, X2: draw + awayWin, '12': decisive },
    asian_handicap_0: { home: homeWin / decisive, away: awayWin / decisive },
  }
}

// --------------------------------------------------------------------------- Elo
/** World-Football-Elo style ratings with a goal-difference multiplier. */
export class EloRating {
  k: number
  homeAdvantage: number
  initial: number
  ratings = new Map<string, number>()

  constructor(k = 20, homeAdvantage = 65, initial = 1500) {
    this.k = k
    this.homeAdvantage = homeAdvantage
    this.initial = initial
  }

  get(team: string): number {
    return this.ratings.get(team) ?? this.initial
  }

  /** Home expected score (win = 1, draw = 0.5) with the home-advantage bonus added. */
  expectedScore(ratingHome: number, ratingAway: number): number {
    return 1 / (1 + 10 ** (-(ratingHome + this.homeAdvantage - ratingAway) / 400))
  }

  /** Zero-sum update: delta = k * G * (W - We); G = 1 for a 1-goal margin, 1.5 for 2, (11 + margin) / 8 for 3+. */
  update(home: string, away: string, homeGoals: number, awayGoals: number): [number, number] {
    const rHome = this.get(home), rAway = this.get(away)
    const margin = Math.abs(homeGoals - awayGoals)
    const g = margin <= 1 ? 1 : margin === 2 ? 1.5 : (11 + margin) / 8
    const result = homeGoals > awayGoals ? 1 : homeGoals === awayGoals ? 0.5 : 0
    const delta = this.k * g * (result - this.expectedScore(rHome, rAway))
    this.ratings.set(home, rHome + delta)
    this.ratings.set(away, rAway - delta)
    return [rHome + delta, rAway - delta]
  }

  /** Reset and replay matches chronologically. */
  fit(matches: readonly MatchRecord[]): Map<string, number> {
    this.ratings = new Map()
    for (const m of [...matches].sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0))) {
      this.update(m.home, m.away, m.home_goals, m.away_goals)
    }
    return this.ratings
  }
}

// --------------------------------------------------------------------------- profile-based predictor
const clampLambda = (v: number) => Math.min(Math.max(v, LAMBDA_MIN), LAMBDA_MAX)

/** Expected goals (home, away): stats lambdas from xG/possession blended with an Elo-implied goal ratio, clamped to [0.05, 6]. */
export function expectedGoalsFromProfiles(
  home: TeamProfile, away: TeamProfile, leagueAvgGoals = 1.35, homeAdvantage = 1.12, eloWeight = 0.5,
): [number, number] {
  if (leagueAvgGoals <= 0 || homeAdvantage <= 0 || !(eloWeight >= 0 && eloWeight <= 1)) {
    throw new Error('need league_avg_goals > 0, home_advantage > 0 and 0 <= elo_weight <= 1')
  }
  for (const t of [home, away]) {
    if (t.xg_for <= 0 || t.xg_against <= 0 || t.defensive_rating <= 0) {
      throw new Error(`xg_for, xg_against and defensive_rating must be > 0 (${t.name})`)
    }
  }
  const league = leagueAvgGoals
  const tilt = POSSESSION_COEF * (home.possession - away.possession)
  const logLam = Math.log(league * (home.xg_for / league) * (away.xg_against / (league * away.defensive_rating)) * homeAdvantage) + tilt
  const logMu = Math.log(league * (away.xg_for / league) * (home.xg_against / (league * home.defensive_rating)) / homeAdvantage) - tilt

  const logTotal = (logLam + logMu) / 2
  const ratioStats = logLam - logMu
  const ratioElo = 2 * Math.log(homeAdvantage) + (home.elo - away.elo) / ELO_LOG_SCALE
  const ratio = (1 - eloWeight) * ratioStats + eloWeight * ratioElo
  return [clampLambda(Math.exp(logTotal + ratio / 2)), clampLambda(Math.exp(logTotal - ratio / 2))]
}

/** What-if adjustments on top of the profile lambdas (form, absences, home-edge multiplier); result clamped to [0.05, 6]. */
export function applyAdjustments(lam: number, mu: number, adj: FootballAdjustments): [number, number] {
  const l = lam * (1 + adj.home_form) * (1 - 0.04 * adj.home_absences) * (1 + 0.03 * adj.away_absences) * adj.home_advantage
  const m = mu * (1 + adj.away_form) * (1 - 0.04 * adj.away_absences) * (1 + 0.03 * adj.home_absences) / adj.home_advantage
  return [clampLambda(l), clampLambda(m)]
}

export interface PredictOptions {
  leagueAvgGoals?: number
  rho?: number
  lam3?: number
  maxGoals?: number
  ouLines?: readonly number[]
  adjustments?: FootballAdjustments
}

export function predictMatch(home: TeamProfile, away: TeamProfile, opts: PredictOptions = {}): FootballPrediction {
  const { leagueAvgGoals = 1.35, rho = -0.08, maxGoals = 10, ouLines = DEFAULT_OU_LINES, adjustments } = opts
  let [lam, mu] = expectedGoalsFromProfiles(home, away, leagueAvgGoals)
  if (adjustments) [lam, mu] = applyAdjustments(lam, mu, adjustments)
  const rawLam3 = opts.lam3 ?? 0
  if (rawLam3 < 0) throw new Error(`lam3 must be >= 0, got ${rawLam3}`)
  const lam3 = Math.min(rawLam3, 0.5 * Math.min(lam, mu)) // keep lam1, lam2 = lam - lam3, mu - lam3 comfortably positive
  const [lo, hi] = rhoBounds(lam, mu)
  const effRho = Math.min(Math.max(rho, lo), hi)
  const matrix = dixonColesMatrix(lam, mu, effRho, lam3, maxGoals)
  return {
    home: home.name, away: away.name, lambda_home: lam, lambda_away: mu, rho: effRho, lam3,
    elo_home: home.elo, elo_away: away.elo,
    elo_win_expectancy: new EloRating().expectedScore(home.elo, away.elo),
    matrix, markets: marketProbabilities(matrix, ouLines),
  }
}

// --------------------------------------------------------------------------- walk-forward backtest feed
/** Leak-free predictions: each match is predicted from Elo + EWMA xG built only from earlier matches; the first `warmup` matches only update state. */
export function walkForwardPredictions(matches: readonly MatchRecord[], warmup = 30, leagueAvgGoals?: number): WalkForwardRow[] {
  const elo = new EloRating()
  const form = new Map<string, [number, number]>() // team -> [xg_for, xg_against]
  let leagueSum = 0, leagueN = 0
  const rows: WalkForwardRow[] = []

  const profile = (team: string, league: number): TeamProfile => {
    const [xg_for, xg_against] = form.get(team) ?? [league, league]
    return teamProfile({ name: team, elo: elo.get(team), xg_for, xg_against })
  }

  const sorted = [...matches].sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0))
  sorted.forEach((m, i) => {
    const homeXg = m.home_xg ?? m.home_goals
    const awayXg = m.away_xg ?? m.away_goals

    if (i >= warmup) {
      const league = leagueAvgGoals || (leagueN ? leagueSum / leagueN : 1.35)
      const mk = predictMatch(profile(m.home, league), profile(m.away, league), { leagueAvgGoals: league, ouLines: [2.5] }).markets
      rows.push({
        date: m.date, home: m.home, away: m.away, home_goals: m.home_goals, away_goals: m.away_goals,
        p_home: mk.home_win, p_draw: mk.draw, p_away: mk.away_win,
        p_over_2_5: mk.over_under['2.5'].over, p_btts: mk.btts_yes,
        outcome: m.home_goals > m.away_goals ? 'H' : m.home_goals === m.away_goals ? 'D' : 'A',
        odds_home: m.odds_home ?? null, odds_draw: m.odds_draw ?? null, odds_away: m.odds_away ?? null,
      })
    }

    // Update state with the result only after predicting it.
    for (const [team, scored, conceded] of [[m.home, homeXg, awayXg], [m.away, awayXg, homeXg]] as const) {
      const old = form.get(team) ?? [scored, conceded]
      form.set(team, [old[0] + EWMA_ALPHA * (scored - old[0]), old[1] + EWMA_ALPHA * (conceded - old[1])])
    }
    elo.update(m.home, m.away, m.home_goals, m.away_goals)
    leagueSum += homeXg + awayXg
    leagueN += 2
  })
  return rows
}
