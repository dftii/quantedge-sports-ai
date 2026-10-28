// Port of backend/app/engines/quant_engine.py: Monte Carlo simulation, odds maths (vig removal, EV, Kelly) and backtesting.
import type {
  BacktestResult, FightSimulationSummary, FootballSimulationSummary, MethodProbs, ValueBet,
} from './types'

export const WILSON_Z = 1.96 // 95% interval
export const MAX_SIMULATIONS = 1_000_000
const CALIBRATION_BINS = 10
const LOG_LOSS_EPS = 1e-15

const checkSimulations = (n: number) => {
  if (!Number.isInteger(n) || n < 1 || n > MAX_SIMULATIONS) throw new Error(`n_simulations must be between 1 and ${MAX_SIMULATIONS}`)
}
const checkProb = (p: number, name = 'probability') => {
  if (!(p >= 0 && p <= 1)) throw new Error(`${name} must be in [0, 1], got ${p}`)
}
const checkOdds = (odds: number) => {
  if (!(odds > 1)) throw new Error(`decimal odds must be > 1.0, got ${odds}`)
}

// --- Random numbers -----------------------------------------------------------------------
/** mulberry32: small, fast, seedable PRNG returning floats in [0, 1). Seeds do NOT reproduce NumPy's default_rng stream. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Numeric seed -> reproducible stream; null -> seeded from crypto (Math.random as fallback). */
export function makeRng(seed: number | null): () => number {
  if (typeof seed === 'number') return mulberry32(seed)
  const c = globalThis.crypto
  return mulberry32(c?.getRandomValues ? c.getRandomValues(new Uint32Array(1))[0] : Math.random() * 4294967296)
}

/** Cumulative distribution (last entry exactly 1) for inverse-transform sampling. */
function cumulative(weights: ArrayLike<number>): Float64Array {
  const cdf = new Float64Array(weights.length)
  let acc = 0
  for (let i = 0; i < weights.length; i++) cdf[i] = acc += weights[i]
  if (!(acc > 0)) throw new Error('probabilities must be non-negative with positive total')
  for (let i = 0; i < cdf.length; i++) cdf[i] /= acc
  cdf[cdf.length - 1] = 1
  return cdf
}

/** First index whose cumulative probability exceeds u (searchsorted side="right"). */
function pick(cdf: Float64Array, u: number): number {
  let lo = 0, hi = cdf.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (cdf[mid] > u) hi = mid
    else lo = mid + 1
  }
  return lo
}

/** 95% Wilson score interval for a binomial proportion. */
export function wilson(successes: number, n: number): [number, number] {
  const p = successes / n
  const z2 = WILSON_Z ** 2
  const denom = 1 + z2 / n
  const centre = (p + z2 / (2 * n)) / denom
  const half = (WILSON_Z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom
  return [Math.max(0, centre - half), Math.min(1, centre + half)]
}

const clampUpTo = (upTo: number | undefined, n: number) => Math.min(n, Math.max(1, Math.floor(upTo ?? n)))

// --- Football Monte Carlo -----------------------------------------------------------------
export interface FootballSimulation {
  n_simulations: number
  elapsed_ms: number
  homeGoals: Int32Array
  awayGoals: Int32Array
  /** Summary of the first `upTo` draws (default all), so a UI can animate partial counts without re-running. */
  summarise(upTo?: number, ouLines?: readonly number[]): FootballSimulationSummary
}

/** Sample scorelines from the exact joint distribution (cumulative table + binary search over the flattened matrix). */
export function simulateFootball(scoreMatrix: number[][], nSimulations = 10_000, seed: number | null = null): FootballSimulation {
  checkSimulations(nSimulations)
  const cols = scoreMatrix[0]?.length ?? 0
  const flat = scoreMatrix.flat()
  if (!cols || scoreMatrix.some((r) => r.length !== cols) || flat.some((v) => !Number.isFinite(v) || v < 0) || !(flat.reduce((s, v) => s + v, 0) > 0)) {
    throw new Error('score_matrix must be a 2-D, finite, non-negative array with positive sum')
  }
  const start = performance.now()
  const rng = makeRng(seed)
  const cdf = cumulative(flat)
  const homeGoals = new Int32Array(nSimulations), awayGoals = new Int32Array(nSimulations)
  for (let s = 0; s < nSimulations; s++) {
    const cell = pick(cdf, rng())
    homeGoals[s] = Math.floor(cell / cols)
    awayGoals[s] = cell % cols
  }
  const elapsed_ms = performance.now() - start

  return {
    n_simulations: nSimulations, elapsed_ms, homeGoals, awayGoals,
    summarise(upTo, ouLines = [0.5, 1.5, 2.5, 3.5, 4.5]) {
      const n = clampUpTo(upTo, nSimulations)
      let wins = 0, draws = 0, losses = 0, btts = 0, sumH = 0, sumA = 0, maxH = 0, maxA = 0
      const overs = ouLines.map(() => 0), unders = ouLines.map(() => 0)
      for (let s = 0; s < n; s++) {
        const h = homeGoals[s], a = awayGoals[s], total = h + a
        if (h > a) wins++
        else if (h === a) draws++
        else losses++
        if (h > 0 && a > 0) btts++
        sumH += h
        sumA += a
        if (h > maxH) maxH = h
        if (a > maxA) maxA = a
        ouLines.forEach((line, l) => { if (total > line) overs[l]++; else if (total < line) unders[l]++ })
      }
      const meanH = sumH / n, meanA = sumA / n
      let varH = 0, varA = 0
      const homeHist = new Array<number>(Math.max(maxH, maxA) + 1).fill(0)
      const awayHist = new Array<number>(homeHist.length).fill(0)
      const width = maxA + 1
      const counts = new Map<number, number>()
      for (let s = 0; s < n; s++) {
        const h = homeGoals[s], a = awayGoals[s]
        varH += (h - meanH) ** 2
        varA += (a - meanA) ** 2
        homeHist[h]++
        awayHist[a]++
        counts.set(h * width + a, (counts.get(h * width + a) ?? 0) + 1)
      }
      const over_under: FootballSimulationSummary['over_under'] = {}
      ouLines.forEach((line, l) => { over_under[String(line)] = { over: overs[l] / n, under: unders[l] / n } })
      const score_distribution = [...counts].sort((x, y) => y[1] - x[1] || x[0] - y[0]).slice(0, 10)
        .map(([i, c]) => ({ score: `${Math.floor(i / width)}-${i % width}`, frequency: c / n }))
      return {
        n_simulations: n, elapsed_ms, home_win: wins / n, draw: draws / n, away_win: losses / n, btts_yes: btts / n,
        over_under, mean_home_goals: meanH, mean_away_goals: meanA,
        std_home_goals: Math.sqrt(varH / n), std_away_goals: Math.sqrt(varA / n),
        confidence_intervals: { home_win: wilson(wins, n), draw: wilson(draws, n), away_win: wilson(losses, n) },
        score_distribution, goal_histogram: { home: homeHist, away: awayHist },
      }
    },
  }
}

// --- Fight Monte Carlo --------------------------------------------------------------------
const METHOD_KEYS = ['ko_tko', 'submission', 'decision'] as const

export interface FightSimulation {
  n_simulations: number
  elapsed_ms: number
  rounds: number
  winners: Uint8Array // 0 = a, 1 = b
  methods: Uint8Array // 0 = ko_tko, 1 = submission, 2 = decision
  endRounds: Uint8Array // decisions = scheduled rounds
  summarise(upTo?: number): FightSimulationSummary
}

/** Sample (winner, method) among the 6 categories, then a round for each finish (numbered-round entries only, renormalised). */
export function simulateFight(
  methodProbs: { a: MethodProbs; b: MethodProbs }, finishRoundProbs: Record<string, number>,
  rounds: number, nSimulations = 10_000, seed: number | null = null,
): FightSimulation {
  checkSimulations(nSimulations)
  if (rounds < 1) throw new Error('rounds must be >= 1')
  const categories = (['a', 'b'] as const).flatMap((s) => METHOD_KEYS.map((k) => methodProbs[s][k]))
  const roundP = Array.from({ length: rounds }, (_, r) => finishRoundProbs[String(r + 1)] ?? 0)
  const finishMass = categories[0] + categories[1] + categories[3] + categories[4]
  if (categories.some((v) => v < 0) || roundP.some((v) => v < 0) || !(categories.reduce((s, v) => s + v, 0) > 0)) {
    throw new Error('probabilities must be non-negative with positive total')
  }
  if (!(roundP.reduce((s, v) => s + v, 0) > 0) && finishMass > 0) {
    throw new Error('finish_round_probs has no mass on numbered rounds but finishes are possible')
  }

  const start = performance.now()
  const rng = makeRng(seed)
  const catCdf = cumulative(categories)
  const roundCdf = finishMass > 0 ? cumulative(roundP) : catCdf
  const winners = new Uint8Array(nSimulations), methods = new Uint8Array(nSimulations)
  const endRounds = new Uint8Array(nSimulations).fill(rounds)
  for (let s = 0; s < nSimulations; s++) {
    const cat = pick(catCdf, rng())
    winners[s] = cat < 3 ? 0 : 1
    methods[s] = cat % 3
    if (methods[s] < 2) endRounds[s] = pick(roundCdf, rng()) + 1
  }
  const elapsed_ms = performance.now() - start

  return {
    n_simulations: nSimulations, elapsed_ms, rounds, winners, methods, endRounds,
    summarise(upTo) {
      const n = clampUpTo(upTo, nSimulations)
      let aWins = 0, finished = 0, decisions = 0
      const side = [0, 1].map(() => [0, 0, 0])
      const roundCounts = new Array<number>(rounds + 1).fill(0)
      for (let s = 0; s < n; s++) {
        if (winners[s] === 0) aWins++
        side[winners[s]][methods[s]]++
        if (methods[s] < 2) { finished++; roundCounts[endRounds[s]]++ } else decisions++
      }
      const toProbs = (c: number[]): MethodProbs => ({ ko_tko: c[0] / n, submission: c[1] / n, decision: c[2] / n })
      const round_distribution: Record<string, number> = {}
      for (let r = 1; r <= rounds; r++) round_distribution[String(r)] = roundCounts[r] / n
      round_distribution.decision = decisions / n
      return {
        n_simulations: n, elapsed_ms, win_prob_a: aWins / n, win_prob_b: 1 - aWins / n,
        confidence_intervals: { a: wilson(aWins, n) }, methods: { a: toProbs(side[0]), b: toProbs(side[1]) },
        round_distribution, finish_rate: finished / n,
      }
    },
  }
}

// --- Odds maths ---------------------------------------------------------------------------
export function impliedProbability(decimalOdds: number): number {
  checkOdds(decimalOdds)
  return 1 / decimalOdds
}

/** Fair probabilities from one mutually exclusive market. proportional: p_i / sum(p); power: p_i^k with sum = 1 (bisection). */
export function removeVig(decimalOdds: readonly number[], method: 'proportional' | 'power' = 'proportional'): number[] {
  if (decimalOdds.length < 2) throw new Error('a market needs at least two selections')
  const p = decimalOdds.map(impliedProbability)
  if (method === 'proportional') {
    const sum = p.reduce((s, v) => s + v, 0)
    return p.map((v) => v / sum)
  }
  if (method === 'power') {
    const excess = (k: number) => p.reduce((s, v) => s + v ** k, 0) - 1
    let lo = 0, hi = 2
    while (excess(hi) > 0) hi *= 2 // decreasing in k; widen until the root is bracketed
    for (let i = 0; i < 200 && hi - lo > 1e-15; i++) {
      const mid = (lo + hi) / 2
      if (excess(mid) > 0) lo = mid
      else hi = mid
    }
    const k = (lo + hi) / 2
    return p.map((v) => v ** k)
  }
  throw new Error("method must be 'proportional' or 'power'")
}

export function bookmakerMargin(decimalOdds: readonly number[]): number {
  return decimalOdds.reduce((s, o) => s + impliedProbability(o), 0) - 1
}

/** Expected profit per unit staked: p * odds - 1. */
export function expectedValue(prob: number, decimalOdds: number): number {
  checkProb(prob)
  checkOdds(decimalOdds)
  return prob * decimalOdds - 1
}

/** Fractional Kelly stake as a share of bankroll: (b*p - q)/b * multiplier, in [0, cap]. */
export function kellyFraction(prob: number, decimalOdds: number, multiplier = 0.25, cap = 0.1): number {
  checkProb(prob)
  checkOdds(decimalOdds)
  const b = decimalOdds - 1
  const full = (b * prob - (1 - prob)) / b
  return Math.min(cap, Math.max(0, full * multiplier))
}

/** Evaluate one mutually exclusive market; selections are [label, modelProb, decimalOdds]. Best expected value first. */
export function evaluateMarket(
  selections: readonly (readonly [string, number, number])[], bankroll = 1000, kellyMultiplier = 0.25, minEv = 0,
): ValueBet[] {
  if (!(bankroll > 0)) throw new Error('bankroll must be > 0')
  const fair = removeVig(selections.map((s) => s[2]))
  return selections.map(([selection, prob, odds], i): ValueBet => {
    const ev = expectedValue(prob, odds)
    const kelly = kellyFraction(prob, odds, kellyMultiplier)
    return {
      selection, model_prob: prob, decimal_odds: odds, implied_prob: impliedProbability(odds), fair_prob: fair[i],
      edge: prob - fair[i], expected_value: ev, kelly_fraction: kelly, stake: bankroll * kelly, is_value: ev > minEv,
    }
  }).sort((a, b) => b.expected_value - a.expected_value)
}

// --- Backtesting --------------------------------------------------------------------------
export interface BacktestRecord {
  probs: Record<string, number>
  odds: Record<string, number | null | undefined>
  outcome: string
}
export type BacktestReport = Omit<BacktestResult, 'n_matches' | 'warmup' | 'predictions'>

/** Score probabilities (every record) and simulate best-bet fractional-Kelly staking on the current bankroll where odds exist for all outcomes. */
export function runBacktest(
  records: readonly BacktestRecord[], outcomeKeys: readonly string[], bankroll = 1000, kellyMultiplier = 0.25, minEv = 0.02,
): BacktestReport {
  if (!records.length) throw new Error('records must not be empty')
  const keys = [...outcomeKeys]
  const probs = records.map((r) => keys.map((k) => r.probs[k]))
  if (probs.some((row) => row.some((p) => !(p >= 0 && p <= 1)))) throw new Error('probabilities must be in [0, 1]')
  const outcomes = records.map((r) => {
    const i = keys.indexOf(r.outcome)
    if (i < 0) throw new Error(`unknown outcome '${r.outcome}'`)
    return i
  })

  const mean = (xs: number[]) => xs.reduce((s, v) => s + v, 0) / xs.length
  const brier = mean(probs.map((row, i) => row.reduce((s, p, k) => s + (p - (k === outcomes[i] ? 1 : 0)) ** 2, 0)))
  const log_loss = -mean(probs.map((row, i) => Math.log(Math.max(row[outcomes[i]], LOG_LOSS_EPS))))
  const accuracy = mean(probs.map((row, i) => (row.indexOf(Math.max(...row)) === outcomes[i] ? 1 : 0)))

  const bins = Array.from({ length: CALIBRATION_BINS }, () => ({ count: 0, sumP: 0, sumY: 0 }))
  probs.forEach((row, i) => row.forEach((p, k) => {
    const b = bins[Math.min(Math.floor(p * CALIBRATION_BINS), CALIBRATION_BINS - 1)]
    b.count++
    b.sumP += p
    b.sumY += k === outcomes[i] ? 1 : 0
  }))
  const calibration = bins.map((b, i) => ({
    bin_low: i / CALIBRATION_BINS, bin_high: (i + 1) / CALIBRATION_BINS,
    mean_predicted: b.count ? b.sumP / b.count : null, observed_frequency: b.count ? b.sumY / b.count : null, count: b.count,
  }))

  let balance = bankroll, stakedTotal = 0
  const curve = [balance]
  const bets: BacktestReport['bets'] = []
  const returns: number[] = []
  records.forEach((r, i) => {
    const odds = keys.map((k) => r.odds[k])
    if (odds.some((o) => o == null)) return
    const best = evaluateMarket(keys.map((k, j) => [k, probs[i][j], odds[j] as number] as const), balance, kellyMultiplier, minEv)[0]
    if (!best.is_value || best.stake <= 0) return
    const profit = best.selection === r.outcome ? best.stake * (best.decimal_odds - 1) : -best.stake
    balance += profit
    stakedTotal += best.stake
    returns.push(profit / best.stake)
    curve.push(balance)
    bets.push({ index: i, selection: best.selection, model_prob: best.model_prob, odds: best.decimal_odds, stake: best.stake, profit, bankroll: balance })
  })

  let peak = curve[0], max_drawdown = 0
  for (const c of curve) {
    peak = Math.max(peak, c)
    max_drawdown = Math.max(max_drawdown, (peak - c) / peak)
  }
  const profit = balance - bankroll
  const roi = stakedTotal > 0 ? profit / stakedTotal : 0
  const mu = returns.length ? mean(returns) : 0
  const std = returns.length >= 2 ? Math.sqrt(returns.reduce((s, v) => s + (v - mu) ** 2, 0) / (returns.length - 1)) : 0
  const sharpe = std > 0 ? (mu / std) * Math.sqrt(returns.length) : 0

  return {
    n_events: records.length, n_bets: bets.length, brier_score: brier, log_loss, accuracy, calibration, roi,
    yield_pct: roi * 100, profit, final_bankroll: balance, max_drawdown, sharpe, bankroll_curve: curve, bets,
  }
}
