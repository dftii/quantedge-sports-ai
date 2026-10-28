import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { Play } from 'lucide-react'
import type { Dataset, Fighter, FightSimulationRequest, FightSimulationResult } from '../engine/types'
import { RunDistribution } from './charts/Histogram'
import { toCounts } from './charts/useReplay'
import { fmtDate, fmtInt, fmtNum, fmtPct, fmtSigned, parseNum } from './format'
import { edge, Tape, type TapeRow } from './Tape'
import { Button } from './ui/Button'
import { NumberField } from './ui/NumberField'
import { Select } from './ui/Select'
import { Slider } from './ui/Slider'
import { EmptyState, Region, Skeleton } from './ui/States'
import { StatFigure } from './ui/StatFigure'
import type { ParamPatch } from './useHashParams'
import type { Runner } from './useRunner'

type FightRunner = Runner<FightSimulationRequest, FightSimulationResult>

interface Form {
  a: number; b: number; rounds: 3 | 5
  formA: number; formB: number
  oA: string; oB: string; oddsNote: string
  bankroll: string; kelly: string; seed: string; n: string
}
type Errors = Partial<Record<'b' | 'oA' | 'oB' | 'bankroll' | 'kelly' | 'seed' | 'n', string>>

const DEBOUNCE_MS = 250

function rivals(fighters: Fighter[], a: number): Fighter[] {
  const wc = fighters.find((f) => f.id === a)?.weight_class
  return fighters.filter((f) => f.id !== a && f.weight_class === wc)
}

function matchFor(ds: Dataset, a: number, b: number): Pick<Form, 'oA' | 'oB' | 'oddsNote'> & { rounds: 3 | 5 | null } {
  const evs = ds.events.filter((e) => e.sport === 'mma' && e.odds_fighter_a != null && e.odds_fighter_b != null
    && ((e.fighter_a_id === a && e.fighter_b_id === b) || (e.fighter_a_id === b && e.fighter_b_id === a)))
  const next = evs.filter((e) => e.status === 'scheduled').sort((x, y) => x.event_date.localeCompare(y.event_date))[0]
  const last = evs.filter((e) => e.status === 'finished').sort((x, y) => y.event_date.localeCompare(x.event_date))[0]
  const ev = next ?? last
  if (!ev) return { oA: '', oB: '', rounds: null, oddsNote: 'Sin cuotas para este cruce en el dataset. Introdúcelas para activar el escáner +EV.' }
  const same = ev.fighter_a_id === a
  return {
    oA: fmtNum((same ? ev.odds_fighter_a : ev.odds_fighter_b)!),
    oB: fmtNum((same ? ev.odds_fighter_b : ev.odds_fighter_a)!),
    rounds: ev.scheduled_rounds === 5 ? 5 : 3,
    oddsNote: next ? `Cuotas del combate programado el ${fmtDate(ev.event_date)}.` : `Cuotas del último cruce, el ${fmtDate(ev.event_date)}.`,
  }
}

function build(f: Form): { req: FightSimulationRequest | null; errors: Errors } {
  const errors: Errors = {}
  const oa = parseNum(f.oA), ob = parseNum(f.oB)
  const typed = [f.oA, f.oB].filter((s) => s.trim() !== '').length
  if (typed > 0) {
    ;([['oA', oa, f.oA], ['oB', ob, f.oB]] as const).forEach(([k, v, raw]) => {
      if (raw.trim() === '') errors[k] = 'Falta esta cuota: o las dos, o ninguna.'
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
      fighter_a_id: f.a, fighter_b_id: f.b, rounds: f.rounds, n_simulations: n!, seed,
      odds: typed === 2 ? { fighter_a: oa!, fighter_b: ob! } : null,
      bankroll: bankroll!, kelly_multiplier: kelly!, adjustments: { form_a: f.formA, form_b: f.formB },
    },
  }
}

function initialForm(ds: Dataset, params: URLSearchParams): Form {
  const ids = ds.fighters.map((f) => f.id)
  const next = ds.events.filter((e) => e.sport === 'mma' && e.status === 'scheduled').sort((x, y) => x.event_date.localeCompare(y.event_date))[0]
  const pick = (v: string | null) => ids.find((id) => String(id) === v)
  let a = pick(params.get('fa')) ?? next?.fighter_a_id ?? ids[0] ?? 0
  if (!ids.includes(a)) a = ids[0] ?? 0
  const opts = rivals(ds.fighters, a)
  const b = opts.find((f) => String(f.id) === params.get('fb'))?.id ?? (next?.fighter_a_id === a ? next.fighter_b_id : null) ?? opts[0]?.id ?? 0
  const m = matchFor(ds, a, b)
  return { a, b, rounds: m.rounds ?? 3, formA: 0, formB: 0, oA: m.oA, oB: m.oB, oddsNote: m.oddsNote, bankroll: '1000', kelly: '0,25', seed: '', n: '10000' }
}

interface Props {
  dataset: Dataset
  runner: FightRunner
  params: URLSearchParams
  setParams: (patch: ParamPatch) => void
}

export function FightSimulator({ dataset, runner, params, setParams }: Props) {
  const fighters = useMemo(() => [...dataset.fighters].sort((x, y) => x.name.localeCompare(y.name, 'es')), [dataset])
  const [form, setForm] = useState<Form>(() => initialForm(dataset, params))
  const { req, errors } = useMemo(() => build(form), [form])
  const set = <K extends keyof Form>(k: K, v: Form[K]) => setForm((f) => ({ ...f, [k]: v }))
  const rivalOptions = useMemo(() => rivals(fighters, form.a), [fighters, form.a])

  const pick = (a: number, b: number) => {
    const m = matchFor(dataset, a, b)
    setForm((f) => ({ ...f, a, b, rounds: m.rounds ?? f.rounds, oA: m.oA, oB: m.oB, oddsNote: m.oddsNote }))
    setParams({ fa: String(a), fb: String(b) })
  }
  const pickA = (a: number) => {
    const opts = rivals(fighters, a)
    pick(a, opts.some((f) => f.id === form.b) ? form.b : (opts[0]?.id ?? form.b))
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

  const slot = runner.data ? 'ready' : runner.status === 'error' ? 'error' : req ? 'loading' : 'empty'
  const label = (f: Fighter) => (f.weight_class ? `${f.name} (${f.weight_class})` : f.name)

  return (
    <div className="workspace">
      <form className="panel controls" onSubmit={launch} aria-label="Parámetros del simulador de MMA">
        <fieldset>
          <legend>Combate</legend>
          <Select label="Luchador A" value={form.a} options={fighters.map((f) => ({ value: f.id, label: label(f) }))} onChange={(v) => pickA(Number(v))} />
          <Select label="Luchador B" value={form.b} options={rivalOptions.map((f) => ({ value: f.id, label: label(f) }))} onChange={(v) => pick(form.a, Number(v))}
            helper="Solo rivales de la misma categoría de peso." error={rivalOptions.length === 0 ? 'No hay rival en esta categoría.' : errors.b} />
          <div className="field">
            <span id="rounds-label" className="label">Asaltos programados</span>
            <div className="seg" role="radiogroup" aria-labelledby="rounds-label">
              {([3, 5] as const).map((r) => (
                <label key={r}>
                  <input type="radio" name="rounds" value={r} checked={form.rounds === r} onChange={() => set('rounds', r)} />
                  <span>{r} asaltos</span>
                </label>
              ))}
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>Ajustes del modelo</legend>
          <Slider label="Estado de forma A" value={form.formA} min={-0.2} max={0.2} step={0.01} onChange={(v) => set('formA', v)} format={(v) => fmtSigned(v)} />
          <Slider label="Estado de forma B" value={form.formB} min={-0.2} max={0.2} step={0.01} onChange={(v) => set('formB', v)} format={(v) => fmtSigned(v)} />
        </fieldset>

        <fieldset>
          <legend>Cuotas decimales</legend>
          <div className="row2">
            <NumberField label="Luchador A" value={form.oA} onChange={(v) => set('oA', v)} error={errors.oA} />
            <NumberField label="Luchador B" value={form.oB} onChange={(v) => set('oB', v)} error={errors.oB} />
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
          Lanzar {req ? fmtInt(req.n_simulations) : '10.000'} simulaciones
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
          {runner.data ? <FightResults result={runner.data} dataset={dataset} runId={runner.runId} /> : null}
        </Region>
      </div>
    </div>
  )
}

function FightResults({ result, dataset, runId }: { result: FightSimulationResult; dataset: Dataset; runId: number }) {
  const { prediction: p, simulation: sim } = result

  const dist = useMemo(() => {
    const rd = sim.round_distribution
    const rounds = Object.keys(rd).filter((k) => k !== 'decision').sort((x, y) => Number(x) - Number(y))
    const keys = [...rounds, ...('decision' in rd ? ['decision'] : [])]
    return {
      categories: keys.map((k) => (k === 'decision' ? 'Decisión' : `Asalto ${k}`)),
      series: [{ name: 'Combates simulados por desenlace', counts: toCounts(keys.map((k) => rd[k]), sim.n_simulations), tone: 'accent' as const }],
    }
  }, [sim])

  const a = dataset.fighters.find((f) => f.name === p.fighter_a)
  const b = dataset.fighters.find((f) => f.name === p.fighter_b)
  const ciA = sim.confidence_intervals.a
  const rows = [
    { name: p.fighter_a, an: p.win_prob_a, mc: sim.win_prob_a, ci: ciA },
    { name: p.fighter_b, an: p.win_prob_b, mc: sim.win_prob_b, ci: [1 - ciA[1], 1 - ciA[0]] as [number, number] },
  ]
  const favourite = Math.max(...rows.map((r) => r.mc))
  const pct = (v: number) => `${fmtNum(v * 100, 0)} %`
  const tape: TapeRow[] | null = a && b ? [
    { label: 'Récord', left: `${a.wins}-${a.losses}-${a.draws}`, right: `${b.wins}-${b.losses}-${b.draws}` },
    { label: 'KO / Sub / Dec', left: `${a.ko_wins}/${a.sub_wins}/${a.dec_wins}`, right: `${b.ko_wins}/${b.sub_wins}/${b.dec_wins}` },
    { label: 'Elo', left: fmtNum(a.elo, 0), right: fmtNum(b.elo, 0), edge: edge(a.elo, b.elo, 'high') },
    { label: 'SLpM', left: fmtNum(a.slpm), right: fmtNum(b.slpm), edge: edge(a.slpm, b.slpm, 'high') },
    { label: 'SApM', left: fmtNum(a.sapm), right: fmtNum(b.sapm), edge: edge(a.sapm, b.sapm, 'low') },
    { label: 'Precisión', left: pct(a.striking_accuracy), right: pct(b.striking_accuracy), edge: edge(a.striking_accuracy, b.striking_accuracy, 'high') },
    { label: 'Defensa de derribo', left: pct(a.td_defense), right: pct(b.td_defense), edge: edge(a.td_defense, b.td_defense, 'high') },
    { label: 'Alcance', left: a.reach_cm ? `${a.reach_cm} cm` : '—', right: b.reach_cm ? `${b.reach_cm} cm` : '—', edge: a.reach_cm && b.reach_cm ? edge(a.reach_cm, b.reach_cm, 'high') : null },
    { label: 'Estilo', left: p.style_a, right: p.style_b },
  ] : null

  return (
    <div className="stack">
      <header className="results__head">
        <h2>{p.fighter_a} contra {p.fighter_b}</h2>
        <p className="muted num">{fmtInt(sim.n_simulations)} simulaciones calculadas en {fmtNum(sim.elapsed_ms, sim.elapsed_ms < 10 ? 1 : 0)} ms</p>
      </header>

      <RunDistribution title="Distribución simulada del desenlace" categories={dist.categories} series={dist.series} total={sim.n_simulations} runKey={runId} />

      <section className="panel" aria-labelledby="fight-win">
        <h3 id="fight-win">Probabilidad de victoria</h3>
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Probabilidad de victoria por luchador">
          <table className="fit">
            <thead>
              <tr><th scope="col">Luchador</th><th scope="col" className="r">Analítica</th><th scope="col" className="r">Monte Carlo</th><th scope="col" className="r">IC 95 %</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name}>
                  <th scope="row">{r.name}</th>
                  <td className="r num">{fmtPct(r.an, 2)}</td>
                  <td className={`r num${r.mc === favourite ? ' is-best' : ''}`}>{fmtPct(r.mc, 2)}</td>
                  <td className="r num muted">{fmtPct(r.ci[0], 1)} a {fmtPct(r.ci[1], 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <dl className="stats">
          <StatFigure label="Asaltos esperados" value={fmtNum(p.expected_rounds)} />
          <StatFigure label="Tasa de finalización" value={fmtPct(sim.finish_rate, 1)} note="combates que no llegan a decisión" />
        </dl>
      </section>

      {tape ? (
        <section className="panel" aria-labelledby="fight-tape">
          <h3 id="fight-tape">Cara a cara</h3>
          <Tape leftName={p.fighter_a} rightName={p.fighter_b} rows={tape} caption={`Métricas de ${p.fighter_a} y ${p.fighter_b}`} />
        </section>
      ) : null}
    </div>
  )
}
