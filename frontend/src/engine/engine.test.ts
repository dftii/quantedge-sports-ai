import { describe, expect, it } from 'vitest'
import dataset from '../data/dataset.json'
import { createLocalService } from '../services/localService'
import {
  applyAdjustments, bivariatePoissonMatrix, dixonColesMatrix, EloRating, expectedGoalsFromProfiles, marketProbabilities,
  poissonPmfVector, predictMatch, rhoBounds, teamProfile, walkForwardPredictions,
} from './football'
import { classifyStyle, fighterProfile, predictFight } from './mma'
import {
  bookmakerMargin, evaluateMarket, expectedValue, kellyFraction, mulberry32, removeVig, runBacktest, simulateFight,
  simulateFootball, wilson,
} from './quant'
import type { Dataset, FootballAdjustments } from './types'

const sum = (xs: number[]) => xs.reduce((s, v) => s + v, 0)
const NEUTRAL: FootballAdjustments = { home_advantage: 1, home_form: 0, away_form: 0, home_absences: 0, away_absences: 0 }
const madrid = teamProfile({ name: 'A', elo: 1950, xg_for: 2, xg_against: 0.9, possession: 58 })
const rival = teamProfile({ name: 'B', elo: 1750, xg_for: 1.4, xg_against: 1.3, possession: 48 })

describe('football', () => {
  it('poisson pmf matches the closed form', () => {
    const p = poissonPmfVector(2.5, 10)
    expect(p[3]).toBeCloseTo(Math.exp(-2.5) * 2.5 ** 3 / 6, 14)
    expect(p[0]).toBeCloseTo(Math.exp(-2.5), 14)
    expect(poissonPmfVector(0, 4)).toEqual([1, 0, 0, 0, 0])
  })

  it('score matrices sum to 1 and have the requested marginal means', () => {
    for (const lam3 of [0, 0.2]) {
      const m = dixonColesMatrix(1.6, 1.1, -0.08, lam3, 10)
      expect(Math.abs(sum(m.flat()) - 1)).toBeLessThan(1e-12)
    }
    const bp = bivariatePoissonMatrix(1.2, 0.9, 0.3, 14)
    expect(Math.abs(sum(bp.flat()) - 1)).toBeLessThan(1e-12)
    const mk = marketProbabilities(bp)
    expect(mk.expected_home_goals).toBeCloseTo(1.5, 3)
    expect(mk.expected_away_goals).toBeCloseTo(1.2, 3)
  })

  it('clips rho into its feasible range and rejects impossible inputs', () => {
    const [lo] = rhoBounds(3, 3), [, hi] = rhoBounds(0.5, 0.5)
    expect(dixonColesMatrix(3, 3, -5, 0, 10)).toEqual(dixonColesMatrix(3, 3, lo, 0, 10))
    expect(dixonColesMatrix(0.5, 0.5, 9, 0, 10)).toEqual(dixonColesMatrix(0.5, 0.5, hi, 0, 10))
    expect(() => dixonColesMatrix(1, 1, 0, 1, 10)).toThrow()
    expect(() => dixonColesMatrix(1, 1, 0, 0, 0)).toThrow()
  })

  it('1X2 and derived markets are coherent', () => {
    const { markets: mk } = predictMatch(madrid, rival, { lam3: 0.1 })
    expect(mk.home_win + mk.draw + mk.away_win).toBeCloseTo(1, 12)
    expect(mk.btts_yes + mk.btts_no).toBeCloseTo(1, 12)
    expect(mk.double_chance['1X']).toBeCloseTo(mk.home_win + mk.draw, 14)
    expect(mk.correct_scores).toHaveLength(10)
    expect(mk.correct_scores[0].probability).toBeGreaterThanOrEqual(mk.correct_scores[1].probability)
    expect(Object.keys(mk.over_under)).toEqual(['0.5', '1.5', '2.5', '3.5', '4.5'])
    expect(mk.home_win).toBeGreaterThan(mk.away_win)
  })

  it('elo ratings are zero-sum and favour the winner', () => {
    const elo = new EloRating()
    elo.update('H', 'A', 3, 0)
    expect(elo.get('H') + elo.get('A')).toBeCloseTo(3000, 12)
    // margin 3 -> G = 14/8; equal teams + 65 home bonus
    const we = 1 / (1 + 10 ** (-65 / 400))
    expect(elo.get('H')).toBeCloseTo(1500 + 20 * 1.75 * (1 - we), 12)
  })

  it('adjustments follow the fixed formulas and clamp', () => {
    expect(applyAdjustments(1.5, 1.2, NEUTRAL)).toEqual([1.5, 1.2])
    const adj = { home_advantage: 1.1, home_form: 0.1, away_form: -0.1, home_absences: 2, away_absences: 1 }
    const [l, m] = applyAdjustments(1.5, 1.2, adj)
    expect(l).toBeCloseTo(1.5 * 1.1 * (1 - 0.08) * 1.03 * 1.1, 12)
    expect(m).toBeCloseTo(1.2 * 0.9 * (1 - 0.04) * 1.06 / 1.1, 12)
    expect(applyAdjustments(5.9, 0.06, { ...NEUTRAL, home_advantage: 1.3, home_absences: 5, away_absences: 5, away_form: -0.2 })).toEqual([6, 0.05])
    const [l0] = expectedGoalsFromProfiles(madrid, rival)
    expect(predictMatch(madrid, rival, { adjustments: { ...NEUTRAL, home_form: 0.1 } }).lambda_home).toBeCloseTo(l0 * 1.1, 12)
  })

  it('walk-forward predictions skip the warmup and never use the match being predicted', () => {
    const matches = Array.from({ length: 12 }, (_, i) => ({
      date: `2025-01-${String(i + 1).padStart(2, '0')}`, home: i % 2 ? 'X' : 'Y', away: i % 2 ? 'Y' : 'X', home_goals: i % 4, away_goals: 1,
    }))
    const rows = walkForwardPredictions(matches, 5)
    expect(rows).toHaveLength(7)
    const altered = matches.map((m, i) => (i === 11 ? { ...m, home_goals: 9 } : m))
    expect(walkForwardPredictions(altered, 5).slice(0, 6)).toEqual(rows.slice(0, 6))
    expect(rows[0].p_home + rows[0].p_draw + rows[0].p_away).toBeCloseTo(1, 12)
  })
})

describe('mma', () => {
  const fa = fighterProfile({ name: 'A', slpm: 5, sapm: 3, striking_defense: 0.6, cardio: 0.8, wins: 20, losses: 2, ko_wins: 12, sub_wins: 2, dec_wins: 6, elo: 1700, age: 27 })
  const fb = fighterProfile({ name: 'B', td_avg: 4, td_accuracy: 0.5, td_defense: 0.7, sub_avg: 1.5, cardio: 0.65, age: 36, reach_cm: null })

  it('is antisymmetric under swapping the fighters', () => {
    const ab = predictFight(fa, fb, 3), ba = predictFight(fb, fa, 3)
    expect(ab.win_prob_a + ab.win_prob_b).toBeCloseTo(1, 15)
    expect(ab.win_prob_a).toBeCloseTo(ba.win_prob_b, 12)
    for (const k of Object.keys(ab.factors)) expect(ab.factors[k]).toBeCloseTo(-ba.factors[k], 12)
  })

  it('method and round probabilities sum to 1', () => {
    for (const rounds of [3, 5] as const) {
      const p = predictFight(fa, fb, rounds)
      const all = Object.values(p.method_probs.a).concat(Object.values(p.method_probs.b))
      expect(sum(all)).toBeCloseTo(1, 12)
      expect(sum(Object.values(p.finish_round_probs))).toBeCloseTo(1, 12)
      expect(Object.keys(p.finish_round_probs)).toHaveLength(rounds + 1)
      expect(p.expected_rounds).toBeGreaterThan(0)
      expect(p.expected_rounds).toBeLessThanOrEqual(rounds)
    }
  })

  it('applies defaults, classifies styles and validates', () => {
    expect(fb.reach_cm).toBe(180)
    expect(classifyStyle(fa)).toBe('striker')
    expect(classifyStyle(fb)).toBe('wrestler')
    expect(classifyStyle(fighterProfile({ name: 'C' }))).toBe('balanced')
    expect(() => fighterProfile({ name: 'bad', cardio: 1.2 })).toThrow('cardio')
    expect(() => predictFight(fa, fb, 4 as 3)).toThrow()
  })
})

describe('quant', () => {
  it('ev, kelly and vig removal', () => {
    expect(expectedValue(0.55, 2)).toBeCloseTo(0.1, 14)
    expect(kellyFraction(0.55, 2, 1, 1)).toBeCloseTo(0.1, 14)
    expect(kellyFraction(0.55, 2)).toBeCloseTo(0.025, 14)
    expect(kellyFraction(0.9, 2, 1)).toBe(0.1) // capped
    expect(kellyFraction(0.4, 2)).toBe(0)
    expect(() => expectedValue(0.5, 1)).toThrow()
    const odds = [2.1, 3.4, 3.8]
    for (const method of ['proportional', 'power'] as const) {
      const fair = removeVig(odds, method)
      expect(Math.abs(sum(fair) - 1)).toBeLessThan(1e-12)
    }
    const power = removeVig(odds, 'power'), prop = removeVig(odds)
    expect(power[0]).toBeGreaterThan(prop[0]) // power shrinks long shots more
    expect(bookmakerMargin(odds)).toBeCloseTo(1 / 2.1 + 1 / 3.4 + 1 / 3.8 - 1, 14)
    const bets = evaluateMarket([['a', 0.6, 2], ['b', 0.4, 2]])
    expect(bets[0].selection).toBe('a')
    expect(bets[0].is_value).toBe(true)
    expect(bets[1].is_value).toBe(false)
  })

  it('wilson interval brackets the proportion', () => {
    const [lo, hi] = wilson(50, 100)
    expect(lo).toBeLessThan(0.5)
    expect(hi).toBeGreaterThan(0.5)
  })

  it('football Monte Carlo converges to the matrix and is fast', () => {
    const { matrix, markets } = predictMatch(madrid, rival)
    simulateFootball(matrix, 1000, 0) // warm-up the JIT
    const sim = simulateFootball(matrix, 10_000, 42)
    expect(sim.elapsed_ms).toBeLessThan(50)
    const s = sim.summarise()
    expect(s.home_win).toBeCloseTo(markets.home_win, 1)
    expect(Math.abs(s.home_win - markets.home_win)).toBeLessThan(0.02)
    expect(Math.abs(s.draw - markets.draw)).toBeLessThan(0.02)
    expect(Math.abs(s.over_under['2.5'].over - markets.over_under['2.5'].over)).toBeLessThan(0.02)
    expect(s.home_win + s.draw + s.away_win).toBeCloseTo(1, 12)
    expect(s.score_distribution).toHaveLength(10)
    expect(sum(s.goal_histogram.home)).toBe(10_000)
    expect(s.confidence_intervals.home_win[0]).toBeLessThan(s.home_win)
    expect(sim.summarise(2500).n_simulations).toBe(2500)
    expect(sum(sim.summarise(2500).goal_histogram.away)).toBe(2500)
  })

  it('numeric seeds reproduce, null seeds differ', () => {
    const { matrix } = predictMatch(madrid, rival)
    expect(simulateFootball(matrix, 500, 7).homeGoals).toEqual(simulateFootball(matrix, 500, 7).homeGoals)
    expect(simulateFootball(matrix, 500, null).awayGoals).not.toEqual(simulateFootball(matrix, 500, null).awayGoals)
    expect(mulberry32(1)()).toBe(mulberry32(1)())
    expect(() => simulateFootball(matrix, 0)).toThrow()
    expect(() => simulateFootball([[0, 0]], 10)).toThrow()
  })

  it('fight Monte Carlo converges to the prediction', () => {
    const p = predictFight(fighterProfile({ name: 'A', slpm: 5, elo: 1700 }), fighterProfile({ name: 'B' }), 5)
    const s = simulateFight(p.method_probs, p.finish_round_probs, 5, 20_000, 3).summarise()
    expect(Math.abs(s.win_prob_a - p.win_prob_a)).toBeLessThan(0.02)
    expect(Math.abs(s.round_distribution.decision - p.finish_round_probs.decision)).toBeLessThan(0.02)
    expect(Math.abs(s.methods.a.ko_tko - p.method_probs.a.ko_tko)).toBeLessThan(0.02)
    expect(sum(Object.values(s.round_distribution))).toBeCloseTo(1, 12)
    expect(s.finish_rate + s.round_distribution.decision).toBeCloseTo(1, 12)
  })

  it('backtest metrics on a hand-checked case', () => {
    const rec = (probs: number[], outcome: string, odds: number[] | null) => ({
      probs: { H: probs[0], D: probs[1], A: probs[2] },
      odds: odds ? { H: odds[0], D: odds[1], A: odds[2] } : {},
      outcome,
    })
    const r = runBacktest([
      rec([0.6, 0.25, 0.15], 'H', [2.0, 3.5, 5.0]),
      rec([0.2, 0.3, 0.5], 'D', null),
    ], ['H', 'D', 'A'], 1000, 1, 0)
    expect(r.brier_score).toBeCloseTo(((0.16 + 0.0625 + 0.0225) + (0.04 + 0.49 + 0.25)) / 2, 12)
    expect(r.log_loss).toBeCloseTo(-(Math.log(0.6) + Math.log(0.3)) / 2, 12)
    expect(r.accuracy).toBe(0.5)
    expect(r.n_bets).toBe(1)
    expect(r.bets[0].selection).toBe('H')
    expect(r.final_bankroll).toBeCloseTo(1000 + r.bets[0].stake, 9) // won at 2.0
    expect(r.calibration.reduce((s, b) => s + b.count, 0)).toBe(6)
    expect(r.max_drawdown).toBe(0)
    expect(r.sharpe).toBe(0) // one bet
  })
})

describe('local service', () => {
  const svc = createLocalService(dataset as unknown as Dataset)
  const data = dataset as unknown as Dataset
  const football = {
    home_team_id: 1, away_team_id: 2, n_simulations: 5000, seed: 1, rho: -0.08, lam3: 0.05, max_goals: 10,
    odds: { home: 2.1, draw: 3.5, away: 3.6 }, bankroll: 1000, kelly_multiplier: 0.25, adjustments: NEUTRAL,
  }

  it('simulates football with value bets', async () => {
    const r = await svc.simulateFootball(football)
    expect(r.prediction.home).toBe('Real Madrid')
    expect(r.value_bets?.map((b) => b.selection).sort()).toEqual(['away', 'draw', 'home'])
    expect(r.market_margin).toBeCloseTo(1 / 2.1 + 1 / 3.5 + 1 / 3.6 - 1, 12)
    expect(r.simulation.n_simulations).toBe(5000)
    expect(svc.mode).toBe('local')
  })

  it('simulates fights with fighter-name selections and rejects bad ids', async () => {
    const fight = { fighter_a_id: 1, fighter_b_id: 2, rounds: 5 as const, n_simulations: 2000, seed: null, odds: { fighter_a: 1.6, fighter_b: 2.4 }, bankroll: 500, kelly_multiplier: 0.5, adjustments: { form_a: 0.1, form_b: 0 } }
    const r = await svc.simulateFight(fight)
    expect(r.value_bets?.map((b) => b.selection).sort()).toEqual([r.prediction.fighter_a, r.prediction.fighter_b].sort())
    expect(Object.keys(r.prediction.finish_round_probs)).toHaveLength(6)
    await expect(svc.simulateFight({ ...fight, fighter_b_id: 1 })).rejects.toThrow('distintos')
    await expect(svc.simulateFight({ ...fight, fighter_b_id: 9999 })).rejects.toThrow('no encontrado')
    await expect(svc.simulateFootball({ ...football, away_team_id: 1 })).rejects.toThrow('distintos')
  })

  it('backtests the bundled football history', async () => {
    const r = await svc.backtestFootball({ warmup: 30, bankroll: 1000, kelly_multiplier: 0.25, min_ev: 0.02 })
    const finished = data.events.filter((e) => e.sport === 'football' && e.status === 'finished' && e.home_goals != null).length
    expect(r.n_matches).toBe(finished)
    expect(r.n_events).toBe(finished - 30)
    expect(r.predictions).toHaveLength(finished - 30)
    expect(r.predictions[0].date).toMatch(/^\d{4}-\d{2}-\d{2}$/)
    expect(r.brier_score).toBeGreaterThan(0)
    expect(r.brier_score).toBeLessThan(1)
    await expect(svc.backtestFootball({ warmup: 100000, bankroll: 1000, kelly_multiplier: 0.25, min_ev: 0.02 })).rejects.toThrow('partidos')
  })
})
