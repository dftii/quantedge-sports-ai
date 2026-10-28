// In-browser QuantService: the TypeScript engines running on the bundled dataset (same behaviour as backend/app/main.py).
import { type MatchRecord, predictMatch, teamProfile, walkForwardPredictions } from '../engine/football'
import { fighterProfile, predictFight } from '../engine/mma'
import { bookmakerMargin, evaluateMarket, runBacktest, simulateFight, simulateFootball } from '../engine/quant'
import type {
  BacktestRequest, BacktestResult, Dataset, FightSimulationRequest, FightSimulationResult, FootballSimulationRequest,
  FootballSimulationResult, QuantService, Team, ValueBet,
} from '../engine/types'

const DEFAULT_LEAGUE_GOALS = 1.35
const FORM_TO_ELO = 250 // Elo shift per unit of fight form

export function createLocalService(dataset: Dataset): QuantService {
  const teams = new Map(dataset.teams.map((t) => [t.id, t]))
  const fighters = new Map(dataset.fighters.map((f) => [f.id, f]))
  const competitions = new Map(dataset.competitions.map((c) => [c.id, c]))

  const find = <T>(rows: Map<number, T>, id: number, what: string): T => {
    const row = rows.get(id)
    if (!row) throw new Error(`${what} ${id} no encontrado`)
    return row
  }
  const leagueGoals = (team: Team) => {
    const competition = competitions.get(team.competition_id)
    return competition ? competition.avg_goals_per_match / 2 : DEFAULT_LEAGUE_GOALS
  }

  return {
    mode: 'local',
    loadDataset: async () => dataset,

    async simulateFootball(req: FootballSimulationRequest): Promise<FootballSimulationResult> {
      const home = find(teams, req.home_team_id, 'Equipo'), away = find(teams, req.away_team_id, 'Equipo')
      if (home.id === away.id) throw new Error('El equipo local y el visitante deben ser distintos')
      // defensive_rating is derived from xg_against_avg, so passing it too would count defence twice.
      const profile = (t: Team) => teamProfile({ name: t.name, elo: t.elo, xg_for: t.xg_for_avg, xg_against: t.xg_against_avg, possession: t.possession_avg })
      const prediction = predictMatch(profile(home), profile(away), {
        leagueAvgGoals: leagueGoals(home), rho: req.rho, lam3: req.lam3, maxGoals: req.max_goals, adjustments: req.adjustments,
      })
      const simulation = simulateFootball(prediction.matrix, req.n_simulations, req.seed).summarise()
      let value_bets: ValueBet[] | null = null, market_margin: number | null = null
      if (req.odds) {
        const { home: oh, draw: od, away: oa } = req.odds
        const p = prediction.markets
        value_bets = evaluateMarket([['home', p.home_win, oh], ['draw', p.draw, od], ['away', p.away_win, oa]], req.bankroll, req.kelly_multiplier)
        market_margin = bookmakerMargin([oh, od, oa])
      }
      return { prediction, simulation, value_bets, market_margin }
    },

    async simulateFight(req: FightSimulationRequest): Promise<FightSimulationResult> {
      const a = find(fighters, req.fighter_a_id, 'Peleador'), b = find(fighters, req.fighter_b_id, 'Peleador')
      if (a.id === b.id) throw new Error('El peleador A y el peleador B deben ser distintos')
      const pa = fighterProfile({ ...a, elo: a.elo + FORM_TO_ELO * req.adjustments.form_a })
      const pb = fighterProfile({ ...b, elo: b.elo + FORM_TO_ELO * req.adjustments.form_b })
      const prediction = predictFight(pa, pb, req.rounds)
      const simulation = simulateFight(prediction.method_probs, prediction.finish_round_probs, req.rounds, req.n_simulations, req.seed).summarise()
      let value_bets: ValueBet[] | null = null, market_margin: number | null = null
      if (req.odds) {
        const { fighter_a: oa, fighter_b: ob } = req.odds
        value_bets = evaluateMarket([[a.name, prediction.win_prob_a, oa], [b.name, prediction.win_prob_b, ob]], req.bankroll, req.kelly_multiplier)
        market_margin = bookmakerMargin([oa, ob])
      }
      return { prediction, simulation, value_bets, market_margin }
    },

    async backtestFootball(req: BacktestRequest): Promise<BacktestResult> {
      const names = new Map(dataset.teams.map((t) => [t.id, t.name]))
      const records: MatchRecord[] = dataset.events
        .filter((e) => e.sport === 'football' && e.status === 'finished' && e.home_goals != null && e.away_goals != null)
        .sort((x, y) => (x.event_date < y.event_date ? -1 : x.event_date > y.event_date ? 1 : x.id - y.id))
        .map((e) => ({
          date: e.event_date, home: names.get(e.home_team_id as number) as string, away: names.get(e.away_team_id as number) as string,
          home_goals: e.home_goals as number, away_goals: e.away_goals as number, home_xg: e.home_xg, away_xg: e.away_xg,
          odds_home: e.odds_home, odds_draw: e.odds_draw, odds_away: e.odds_away,
        }))
      if (records.length <= req.warmup) throw new Error(`Se necesitan más de ${req.warmup} partidos terminados, hay ${records.length}`)
      const rows = walkForwardPredictions(records, req.warmup)
      const report = runBacktest(
        rows.map((r) => ({
          probs: { H: r.p_home, D: r.p_draw, A: r.p_away },
          odds: { H: r.odds_home, D: r.odds_draw, A: r.odds_away },
          outcome: r.outcome,
        })),
        ['H', 'D', 'A'], req.bankroll, req.kelly_multiplier, req.min_ev,
      )
      return { ...report, n_matches: records.length, warmup: req.warmup, predictions: rows }
    },
  }
}
