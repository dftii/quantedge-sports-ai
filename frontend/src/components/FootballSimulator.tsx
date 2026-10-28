import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { Play } from 'lucide-react'
import type { Dataset, FootballSimulationRequest, FootballSimulationResult } from '../engine/types'
import { RunDistribution } from './charts/Histogram'
import { toCounts } from './charts/useReplay'
import { fmtDate, fmtInt, fmtNum, fmtPct, fmtSigned, parseNum } from './format'
import { edge, FormGlyphs, Tape, type TapeRow } from './Tape'
import { Button } from './ui/Button'
import { NumberField } from './ui/NumberField'
import { Select } from './ui/Select'
import { Slider } from './ui/Slider'
import { EmptyState, Region, Skeleton } from './ui/States'
import { StatFigure } from './ui/StatFigure'
import type { ParamPatch } from './useHashParams'
import type { Runner } from './useRunner'

type FootballRunner = Runner<FootballSimulationRequest, FootballSimulationResult>

interface Form {
  home: number; away: number
  adv: number; hForm: number; aForm: number; hAbs: number; aAbs: number
  oH: string; oD: string; oA: string; oddsNote: string
  bankroll: string; kelly: string; seed: string; n: string
}
type Errors = Partial<Record<'away' | 'oH' | 'oD' | 'oA' | 'bankroll' | 'kelly' | 'seed' | 'n', string>>

const DEBOUNCE_MS = 250

function oddsFor(ds: Dataset, h: number, a: number): Pick<Form, 'oH' | 'oD' | 'oA' | 'oddsNote'> {
  const withOdds = ds.events.filter(
    (e) => e.sport === 'football' && e.home_team_id === h && e.away_team_id === a && e.odds_home != null && e.odds_draw != null && e.odds_away != null,
  )
  const next = withOdds.filter((e) => e.status === 'scheduled').sort((x, y) => x.event_date.localeCompare(y.event_date))[0]
  const last = withOdds.filter((e) => e.status === 'finished').sort((x, y) => y.event_date.localeCompare(x.event_date))[0]
  const ev = next ?? last
  if (!ev) return { oH: '', oD: '', oA: '', oddsNote: 'Sin cuotas para este cruce en el dataset. Introdúcelas para activar el escáner +EV.' }
  return {
    oH: fmtNum(ev.odds_home!), oD: fmtNum(ev.odds_draw!), oA: fmtNum(ev.odds_away!),
    oddsNote: next ? `Cuotas del partido programado el ${fmtDate(ev.event_date)}.` : `Cuotas del último cruce, el ${fmtDate(ev.event_date)}.`,
  }
}

function build(f: Form): { req: FootballSimulationRequest | null; errors: Errors } {
  const errors: Errors = {}
  if (f.home === f.away) errors.away = 'Elige dos equipos distintos.'
  const oh = parseNum(f.oH), od = parseNum(f.oD), oa = parseNum(f.oA)
  const given = [oh, od, oa].filter((v) => v !== null).length
  const typed = [f.oH, f.oD, f.oA].filter((s) => s.trim() !== '').length
  if (typed > 0) {
    ;([['oH', oh, f.oH], ['oD', od, f.oD], ['oA', oa, f.oA]] as const).forEach(([k, v, raw]) => {
      if (raw.trim() === '') errors[k] = 'Falta esta cuota: o las tres, o ninguna.'
      else if (v === null || v <= 1) errors[k] = 'Cuota decimal mayor que 1,00.'
    })
  }
  const bankroll = parseNum(f.bankroll)
  if (bankroll === null || bankroll <= 0) errors.bankroll = 'Mayor que 0.'
  const kelly = parseNum(f.kelly)
  if (kelly === null || kelly <= 0 || kelly > 1) errors.kelly = 'Entre 0,01 y 1.'
  const n = parseNum(f.n)
  if (n === null || !Number.isInteger(n) || n < 100 || n > 1_000_000) errors.n = 'Entero entre 100 y 1.000.000.'
  const seed = f.seed.trim() === '' ? null : parseNum(f.seed)
  if (f.seed.trim() !== '' && (seed === null || !Number.isInteger(seed) || seed < 0)) errors.seed = 'Entero a partir de 0, o vacío.'
  if (Object.keys(errors).length > 0) return { req: null, errors }
  return {
    errors,
    req: {
      home_team_id: f.home, away_team_id: f.away, n_simulations: n!, seed, rho: -0.08, lam3: 0, max_goals: 10,
      odds: given === 3 ? { home: oh!, draw: od!, away: oa! } : null,
      bankroll: bankroll!, kelly_multiplier: kelly!,
      adjustments: { home_advantage: f.adv, home_form: f.hForm, away_form: f.aForm, home_absences: f.hAbs, away_absences: f.aAbs },
    },
  }
}

function initialForm(ds: Dataset, teams: Dataset['teams'], params: URLSearchParams): Form {
  const byId = (v: string | null) => teams.find((t) => String(t.id) === v)?.id
  const next = ds.events.filter((e) => e.sport === 'football' && e.status === 'scheduled').sort((a, b) => a.event_date.localeCompare(b.event_date))[0]
  const home = byId(params.get('h')) ?? next?.home_team_id ?? teams[0]?.id ?? 0
  const away = byId(params.get('a')) ?? next?.away_team_id ?? teams.find((t) => t.id !== home)?.id ?? 0
  return { home, away, adv: 1, hForm: 0, aForm: 0, hAbs: 0, aAbs: 0, ...oddsFor(ds, home, away), bankroll: '1000', kelly: '0,25', seed: '', n: '10000' }
}

interface Props {
  dataset: Dataset
  runner: FootballRunner
  params: URLSearchParams
  setParams: (patch: ParamPatch) => void
}

export function FootballSimulator({ dataset, runner, params, setParams }: Props) {
  const teams = useMemo(() => dataset.teams.filter((t) => dataset.competitions.find((c) => c.id === t.competition_id)?.sport !== 'mma').sort((a, b) => a.name.localeCompare(b.name, 'es')), [dataset])
  const [form, setForm] = useState<Form>(() => initialForm(dataset, teams, params))
  const { req, errors } = useMemo(() => build(form), [form])
  const set = <K extends keyof Form>(k: K, v: Form[K]) => setForm((f) => ({ ...f, [k]: v }))

  const pickTeam = (side: 'home' | 'away', id: number) => {
    const home = side === 'home' ? id : form.home
    const away = side === 'away' ? id : form.away
    setForm((f) => ({ ...f, home, away, ...oddsFor(dataset, home, away) }))
    setParams({ h: String(home), a: String(away) })
  }

  const timer = useRef<number | undefined>(undefined)
  const first = useRef(true)
  const { run } = runner
  useEffect(() => {
    if (!req) return
    timer.current = window.setTimeout(() => run(req), first.current ? 0 : DEBOUNCE_MS)
    first.current = false
    return () => window.clearTimeout(timer.current)
  }, [req, run])

  const launch = (e?: FormEvent) => {
    e?.preventDefault()
    window.clearTimeout(timer.current)
    if (req) run(req)
  }

  const teamOptions = (other: number) => teams.map((t) => ({ value: t.id, label: t.name, disabled: t.id === other }))
  const slot = runner.data ? 'ready' : runner.status === 'error' ? 'error' : req ? 'loading' : 'empty'
  const nLabel = req ? fmtInt(req.n_simulations) : '10.000'

  return (
    <div className="workspace">
      <form className="panel controls" onSubmit={launch} aria-label="Parámetros del simulador de fútbol">
        <fieldset>
          <legend>Enfrentamiento</legend>
          <Select label="Equipo local" value={form.home} options={teamOptions(form.away)} onChange={(v) => pickTeam('home', Number(v))} />
          <Select label="Equipo visitante" value={form.away} options={teamOptions(form.home)} onChange={(v) => pickTeam('away', Number(v))} error={errors.away} />
        </fieldset>

        <fieldset>
          <legend>Ajustes del modelo</legend>
          <Slider label="Factor cancha" value={form.adv} min={0.8} max={1.3} step={0.01} onChange={(v) => set('adv', v)} format={(v) => `×${fmtNum(v)}`} />
          <Slider label="Estado de forma local" value={form.hForm} min={-0.2} max={0.2} step={0.01} onChange={(v) => set('hForm', v)} format={(v) => fmtSigned(v)} />
          <Slider label="Estado de forma visitante" value={form.aForm} min={-0.2} max={0.2} step={0.01} onChange={(v) => set('aForm', v)} format={(v) => fmtSigned(v)} />
          <Slider label="Bajas clave local" value={form.hAbs} min={0} max={5} step={1} onChange={(v) => set('hAbs', v)} format={(v) => fmtInt(v)} />
          <Slider label="Bajas clave visitante" value={form.aAbs} min={0} max={5} step={1} onChange={(v) => set('aAbs', v)} format={(v) => fmtInt(v)} />
        </fieldset>

        <fieldset>
          <legend>Cuotas decimales</legend>
          <div className="row3">
            <NumberField label="Local" value={form.oH} onChange={(v) => set('oH', v)} error={errors.oH} />
            <NumberField label="Empate" value={form.oD} onChange={(v) => set('oD', v)} error={errors.oD} />
            <NumberField label="Visitante" value={form.oA} onChange={(v) => set('oA', v)} error={errors.oA} />
          </div>
          <p className="note">{form.oddsNote}</p>
        </fieldset>

        <fieldset>
          <legend>Gestión y simulación</legend>
          <div className="row2">
            <NumberField label="Bankroll (unidades)" value={form.bankroll} onChange={(v) => set('bankroll', v)} error={errors.bankroll} />
            <NumberField label="Fracción de Kelly" value={form.kelly} onChange={(v) => set('kelly', v)} error={errors.kelly} />
            <NumberField label="Semilla (opcional)" value={form.seed} onChange={(v) => set('seed', v)} error={errors.seed} placeholder="aleatoria" />
            <NumberField label="Simulaciones" value={form.n} onChange={(v) => set('n', v)} error={errors.n} />
          </div>
        </fieldset>

        <Button type="submit" icon={<Play size={16} aria-hidden="true" />} busy={runner.status === 'loading'} disabled={!req}>
          Lanzar {nLabel} simulaciones
        </Button>
      </form>

      <div className="results">
        <Region
          state={slot}
          error={runner.error}
          onRetry={launch}
          busy={runner.status === 'loading'}
          skeleton={<div className="stack"><Skeleton shape="figures" /><Skeleton shape="chart" /><Skeleton shape="table" /></div>}
          empty={<EmptyState title="Faltan parámetros válidos">Corrige los campos marcados en el panel para lanzar la simulación.</EmptyState>}
        >
          {runner.data ? <FootballResults result={runner.data} dataset={dataset} runId={runner.runId} /> : null}
        </Region>
      </div>
    </div>
  )
}

function FootballResults({ result, dataset, runId }: { result: FootballSimulationResult; dataset: Dataset; runId: number }) {
  const { prediction: p, simulation: sim } = result
  const m = p.markets

  const dist = useMemo(() => {
    const n = sim.n_simulations
    const fold = (arr: number[]) => {
      const c = toCounts(arr, n)
      const head = Array.from({ length: 7 }, (_, i) => c[i] ?? 0)
      return [...head, c.slice(7).reduce((a, b) => a + b, 0)]
    }
    const top = sim.score_distribution.slice(0, 8)
    return {
      series: [
        { name: `${p.home} (local)`, counts: fold(sim.goal_histogram.home), tone: 'accent' as const },
        { name: `${p.away} (visitante)`, counts: fold(sim.goal_histogram.away), tone: 'soft' as const },
      ],
      ranked: { title: 'Marcadores más frecuentes', labels: top.map((s) => s.score), counts: toCounts(top.map((s) => s.frequency), n) },
    }
  }, [sim, p.home, p.away])

  const lines = Object.keys(m.over_under).sort((a, b) => Number(a) - Number(b))
  const home = dataset.teams.find((t) => t.name === p.home)
  const away = dataset.teams.find((t) => t.name === p.away)
  const tape: TapeRow[] | null = home && away ? [
    { label: 'Elo', left: fmtNum(home.elo, 0), right: fmtNum(away.elo, 0), edge: edge(home.elo, away.elo, 'high') },
    { label: 'xG a favor', left: fmtNum(home.xg_for_avg), right: fmtNum(away.xg_for_avg), edge: edge(home.xg_for_avg, away.xg_for_avg, 'high') },
    { label: 'xG en contra', left: fmtNum(home.xg_against_avg), right: fmtNum(away.xg_against_avg), edge: edge(home.xg_against_avg, away.xg_against_avg, 'low') },
    { label: 'Posesión', left: `${fmtNum(home.possession_avg, 0)} %`, right: `${fmtNum(away.possession_avg, 0)} %` },
    { label: 'PPDA', left: fmtNum(home.ppda, 1), right: fmtNum(away.ppda, 1), edge: edge(home.ppda, away.ppda, 'low') },
    { label: 'Forma', left: <FormGlyphs form={home.form} />, right: <FormGlyphs form={away.form} /> },
  ] : null

  const rows = [
    { key: 'home_win', label: 'Local', name: p.home, a: m.home_win, mc: sim.home_win, ci: sim.confidence_intervals.home_win },
    { key: 'draw', label: 'Empate', name: '', a: m.draw, mc: sim.draw, ci: sim.confidence_intervals.draw },
    { key: 'away_win', label: 'Visitante', name: p.away, a: m.away_win, mc: sim.away_win, ci: sim.confidence_intervals.away_win },
  ]
  const favourite = Math.max(...rows.map((r) => r.mc))

  return (
    <div className="stack">
      <header className="results__head">
        <h2>{p.home} contra {p.away}</h2>
        <p className="muted num">{fmtInt(sim.n_simulations)} simulaciones calculadas en {fmtNum(sim.elapsed_ms, sim.elapsed_ms < 10 ? 1 : 0)} ms</p>
      </header>

      <RunDistribution title="Distribución simulada de goles" categories={['0', '1', '2', '3', '4', '5', '6', '7+']}
        series={dist.series} ranked={dist.ranked} total={sim.n_simulations} runKey={runId} />

      <section className="panel" aria-labelledby="h2h-1x2">
        <h3 id="h2h-1x2">Resultado 1X2</h3>
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Probabilidades 1X2">
          <table className="fit">
            <thead>
              <tr><th scope="col">Resultado</th><th scope="col" className="r">Analítica</th><th scope="col" className="r">Monte Carlo</th><th scope="col" className="r">IC 95 %</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.key}>
                  <th scope="row">{r.label}{r.name ? <span className="muted"> {r.name}</span> : null}</th>
                  <td className="r num">{fmtPct(r.a, 2)}</td>
                  <td className={`r num${r.mc === favourite ? ' is-best' : ''}`}>{fmtPct(r.mc, 2)}</td>
                  <td className="r num muted">{fmtPct(r.ci[0], 1)} a {fmtPct(r.ci[1], 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <dl className="stats">
          <StatFigure label="λ local" value={fmtNum(p.lambda_home)} note={`media simulada ${fmtNum(sim.mean_home_goals)}`} />
          <StatFigure label="λ visitante" value={fmtNum(p.lambda_away)} note={`media simulada ${fmtNum(sim.mean_away_goals)}`} />
          <StatFigure label="Ambos marcan" value={fmtPct(sim.btts_yes, 1)} note={`analítica ${fmtPct(m.btts_yes, 1)}`} />
          <StatFigure label="Elo, esperanza local" value={fmtPct(p.elo_win_expectancy, 1)} note={`${fmtNum(p.elo_home, 0)} contra ${fmtNum(p.elo_away, 0)}`} />
        </dl>
      </section>

      <section className="panel" aria-labelledby="h2h-ou">
        <h3 id="h2h-ou">Más y menos goles</h3>
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Líneas de más y menos goles">
          <table className="fit">
            <thead>
              <tr><th scope="col">Línea</th><th scope="col" className="r">Más, analítica</th><th scope="col" className="r">Menos, analítica</th><th scope="col" className="r">Más, Monte Carlo</th><th scope="col" className="r">Menos, Monte Carlo</th></tr>
            </thead>
            <tbody>
              {lines.map((l) => (
                <tr key={l}>
                  <th scope="row" className="num">{fmtNum(Number(l), 1)}</th>
                  <td className="r num">{fmtPct(m.over_under[l].over, 1)}</td>
                  <td className="r num">{fmtPct(m.over_under[l].under, 1)}</td>
                  <td className="r num">{sim.over_under[l] ? fmtPct(sim.over_under[l].over, 1) : '—'}</td>
                  <td className="r num">{sim.over_under[l] ? fmtPct(sim.over_under[l].under, 1) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {tape ? (
        <section className="panel" aria-labelledby="h2h-tape">
          <h3 id="h2h-tape">Cara a cara</h3>
          <Tape leftName={p.home} rightName={p.away} rows={tape} caption={`Métricas de ${p.home} y ${p.away}`} />
        </section>
      ) : null}
    </div>
  )
}
