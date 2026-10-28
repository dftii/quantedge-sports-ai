// Port of backend/app/engines/models_mma.py: multifactor logistic win probability + method/round distribution.
// Every factor is antisymmetric in (a, b), in logit space (positive favours fighter A).
import type { MethodProbs, MMAPrediction } from './types'

// Win-probability weights (logit units)
const W_STRIKING = 0.4
const W_GRAPPLING = 0.5
const W_REACH = 0.25
const W_CARDIO = 0.8 // per unit cardio difference in a 3-round fight; x2 for 5 rounds
const W_AGE = 0.012
const W_EXP_VOLUME = 0.12
const W_EXP_RECORD = 1.2
const W_ELO = 0.6
const W_STYLE = 0.35

const AGE_PRIME = 32
const AGE_MAX_PENALTY = 1.5
const RECORD_PRIOR_FIGHTS = 4

// Reference "average UFC fighter" values
const REF_STRIKING_DEF = 0.55
const REF_STRIKE_RATE = 3.5
const REF_TD_AVG = 1.5
const REF_TD_ACC = 0.4
const REF_TD_DEF = 0.6
const REF_SUB_AVG = 0.5

// Method / round constants
const PRIOR_SHARES = { ko: 0.32, sub: 0.18, dec: 0.5 }
const PRIOR_WINS = 8
const DEC_DEFENSE_W = 3
const DEC_CARDIO_W = 1
const FIVE_ROUND_DEC_SCALE = 0.8
const KO_ROUND_DECAY = 0.3
const SUB_ROUND_GROWTH = 0.1
const CARDIO_FADE = 1.5

const METHODS = ['ko_tko', 'submission', 'decision'] as const

export interface FighterProfile {
  name: string
  slpm: number
  sapm: number
  striking_accuracy: number
  striking_defense: number
  td_avg: number
  td_accuracy: number
  td_defense: number
  sub_avg: number
  reach_cm: number
  height_cm: number
  cardio: number
  age: number
  wins: number
  losses: number
  ko_wins: number
  sub_wins: number
  dec_wins: number
  elo: number
}

const FIGHTER_DEFAULTS: Omit<FighterProfile, 'name'> = {
  slpm: 3.5, sapm: 3.5, striking_accuracy: 0.45, striking_defense: 0.55, td_avg: 1.5, td_accuracy: 0.4, td_defense: 0.6,
  sub_avg: 0.5, reach_cm: 180, height_cm: 178, cardio: 0.7, age: 30, wins: 10, losses: 3, ko_wins: 4, sub_wins: 3,
  dec_wins: 3, elo: 1500,
}

const NON_NEGATIVE = ['slpm', 'sapm', 'td_avg', 'sub_avg', 'wins', 'losses', 'ko_wins', 'sub_wins', 'dec_wins'] as const
const UNIT_RANGE = ['striking_accuracy', 'striking_defense', 'td_accuracy', 'td_defense', 'cardio'] as const
const POSITIVE = ['reach_cm', 'height_cm', 'age'] as const

/** Build a validated profile; missing or null values (e.g. unknown age/height/reach) fall back to the defaults. */
export function fighterProfile(init: { name: string } & { [K in keyof Omit<FighterProfile, 'name'>]?: number | null }): FighterProfile {
  const f: FighterProfile = { name: init.name, ...FIGHTER_DEFAULTS }
  for (const k of Object.keys(FIGHTER_DEFAULTS) as (keyof typeof FIGHTER_DEFAULTS)[]) f[k] = init[k] ?? FIGHTER_DEFAULTS[k]
  for (const k of NON_NEGATIVE) if (!(f[k] >= 0)) throw new Error(`${f.name}: ${k} must be >= 0`)
  for (const k of UNIT_RANGE) if (!(f[k] >= 0 && f[k] <= 1)) throw new Error(`${f.name}: ${k} must be in [0, 1]`)
  for (const k of POSITIVE) if (!(f[k] > 0)) throw new Error(`${f.name}: ${k} must be > 0`)
  return f
}

const strikingIndex = (f: FighterProfile) => f.slpm / REF_STRIKE_RATE

function grapplingIndex(f: FighterProfile): number {
  const ref = REF_TD_AVG * REF_TD_ACC + 0.5 * REF_SUB_AVG
  return (f.td_avg * f.td_accuracy + 0.5 * f.sub_avg) / ref
}

/** "striker" (ratio >= 1.4), "grappler"/"wrestler" (ratio <= 0.7, split at 40% submission share), else "balanced". */
export function classifyStyle(f: FighterProfile): string {
  const grappling = grapplingIndex(f)
  const ratio = grappling > 0 ? strikingIndex(f) / grappling : Infinity
  if (ratio >= 1.4) return 'striker'
  if (ratio <= 0.7) {
    const subShare = (0.5 * f.sub_avg) / (f.td_avg * f.td_accuracy + 0.5 * f.sub_avg)
    return subShare >= 0.4 ? 'grappler' : 'wrestler'
  }
  return 'balanced'
}

// --- Win-probability factors
const strikesLanded = (x: FighterProfile, y: FighterProfile) =>
  0.5 * ((x.slpm * (1 - y.striking_defense)) / (1 - REF_STRIKING_DEF) + y.sapm)

function grapplingThreat(x: FighterProfile, y: FighterProfile): number {
  const takedowns = x.td_avg * x.td_accuracy * (1 - y.td_defense)
  return takedowns + 0.5 * x.sub_avg * (0.3 + Math.tanh(takedowns))
}

const agePenalty = (age: number) => Math.min(AGE_MAX_PENALTY, W_AGE * Math.max(0, age - AGE_PRIME) ** 2)

const recordStrength = (f: FighterProfile) => (f.wins + RECORD_PRIOR_FIGHTS / 2) / (f.wins + f.losses + RECORD_PRIOR_FIGHTS)

function styleEdge(x: FighterProfile, sx: string, y: FighterProfile, sy: string): number {
  if ((sx === 'wrestler' || sx === 'grappler') && sy === 'striker') {
    return W_STYLE * Math.min(1, Math.max(0, (REF_TD_DEF - y.td_defense) / 0.3))
  }
  if (sx === 'striker' && (sy === 'wrestler' || sy === 'grappler')) {
    return W_STYLE * Math.min(1, Math.max(0, (x.td_defense - REF_TD_DEF) / 0.3))
  }
  return 0
}

function factorsOf(a: FighterProfile, b: FighterProfile, rounds: number, styleA: string, styleB: string): Record<string, number> {
  const strikingDiff = strikesLanded(a, b) - strikesLanded(b, a)
  const grapplingDiff = grapplingThreat(a, b) - grapplingThreat(b, a)
  const lengthDiff = a.reach_cm - b.reach_cm + 0.5 * (a.height_cm - b.height_cm)
  const cardioScale = (rounds - 1) / 2
  return {
    striking_differential: W_STRIKING * 3 * Math.tanh(strikingDiff / 3),
    grappling_matchup: W_GRAPPLING * 2 * Math.tanh(grapplingDiff / 2),
    reach_advantage: W_REACH * Math.tanh(lengthDiff / 10),
    cardio: W_CARDIO * cardioScale * (a.cardio - b.cardio),
    age: agePenalty(b.age) - agePenalty(a.age),
    experience: W_EXP_VOLUME * (Math.log1p(a.wins + a.losses) - Math.log1p(b.wins + b.losses))
      + W_EXP_RECORD * (recordStrength(a) - recordStrength(b)),
    elo: ((W_ELO * (a.elo - b.elo)) / 400) * Math.LN10,
    style_matchup: styleEdge(a, styleA, b, styleB) - styleEdge(b, styleB, a, styleA),
  }
}

// --- Method and round distribution
/** Unnormalised [ko, sub, dec] weights for X winning against Y. */
function methodWeights(x: FighterProfile, y: FighterProfile, meanDef: number, meanCardio: number, rounds: number): number[] {
  const n = x.ko_wins + x.sub_wins + x.dec_wins + PRIOR_WINS
  const shares = [
    (x.ko_wins + PRIOR_WINS * PRIOR_SHARES.ko) / n,
    (x.sub_wins + PRIOR_WINS * PRIOR_SHARES.sub) / n,
    (x.dec_wins + PRIOR_WINS * PRIOR_SHARES.dec) / n,
  ]
  const strikingOutput = (x.slpm * (1 - y.striking_defense)) / (REF_STRIKE_RATE * (1 - REF_STRIKING_DEF))
  const koMult = Math.min(2.5, Math.max(0.4, Math.sqrt((strikingOutput * y.sapm) / REF_STRIKE_RATE)))

  const tdSuccess = (x.td_avg * x.td_accuracy * (1 - y.td_defense)) / (REF_TD_AVG * REF_TD_ACC * (1 - REF_TD_DEF))
  const subMult = Math.min(3, Math.max(0.2, Math.sqrt((x.sub_avg / REF_SUB_AVG) * (0.3 + 0.7 * tdSuccess))))

  let decMult = Math.exp(DEC_DEFENSE_W * (meanDef - 0.575) + DEC_CARDIO_W * (meanCardio - 0.7))
  if (rounds === 5) decMult *= FIVE_ROUND_DEC_SCALE
  return [shares[0] * koMult, shares[1] * subMult, shares[2] * decMult]
}

function roundProfile(rounds: number, slope: number): number[] {
  const raw = Array.from({ length: rounds }, (_, r) => Math.exp(slope * r))
  const total = raw.reduce((s, v) => s + v, 0)
  return raw.map((w) => w / total)
}

/** Win probability, method of victory and finish timing for A vs B. */
export function predictFight(a: FighterProfile, b: FighterProfile, rounds: 3 | 5 = 3): MMAPrediction {
  if (rounds !== 3 && rounds !== 5) throw new Error('rounds must be 3 or 5')

  const styleA = classifyStyle(a), styleB = classifyStyle(b)
  const factors = factorsOf(a, b, rounds, styleA, styleB)
  const winA = 1 / (1 + Math.exp(-Object.values(factors).reduce((s, v) => s + v, 0)))
  const winB = 1 - winA

  const meanDef = (a.striking_defense + b.striking_defense + a.td_defense + b.td_defense) / 4
  const meanCardio = (a.cardio + b.cardio) / 2
  const side = (win: number, x: FighterProfile, y: FighterProfile): MethodProbs => {
    const w = methodWeights(x, y, meanDef, meanCardio, rounds)
    const total = w[0] + w[1] + w[2]
    const [ko_tko, submission, decision] = w.map((wi) => (win * wi) / total)
    return { ko_tko, submission, decision }
  }
  const method_probs = { a: side(winA, a, b), b: side(winB, b, a) }

  const total = (m: (typeof METHODS)[number]) => method_probs.a[m] + method_probs.b[m]
  const koTotal = total('ko_tko'), subTotal = total('submission'), decTotal = total('decision')

  const fade = CARDIO_FADE * (0.7 - meanCardio) // > 0 for tired fighters: later finishes
  const koProfile = roundProfile(rounds, -KO_ROUND_DECAY + fade)
  const subProfile = roundProfile(rounds, SUB_ROUND_GROWTH + fade)
  const finishByRound = koProfile.map((k, r) => koTotal * k + subTotal * subProfile[r])

  const finish_round_probs: Record<string, number> = {}
  finishByRound.forEach((p, r) => { finish_round_probs[String(r + 1)] = p })
  finish_round_probs.decision = decTotal
  const expected_rounds = finishByRound.reduce((s, p, r) => s + p * (r + 0.5), 0) + decTotal * rounds

  return {
    fighter_a: a.name, fighter_b: b.name, style_a: styleA, style_b: styleB,
    win_prob_a: winA, win_prob_b: winB, method_probs, expected_rounds, finish_round_probs, factors,
  }
}
