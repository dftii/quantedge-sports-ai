import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { Check, Play } from 'lucide-react'
import type { BacktestRequest, BacktestResult } from '../engine/types'
import { CalibrationChart } from './charts/CalibrationChart'
import { LineChart } from './charts/LineChart'
import { fmtDate, fmtInt, fmtNum, fmtPct, fmtSigned, fmtSignedPct, parseNum } from './format'
import { Button } from './ui/Button'
import { NumberField } from './ui/NumberField'
import { EmptyState, Region, Skeleton, type RegionState } from './ui/States'
import { StatFigure } from './ui/StatFigure'
import type { Runner } from './useRunner'

interface Form { warmup: string; bankroll: string; kelly: string; minEv: string }
type Errors = Partial<Record<keyof Form, string>>

const DEFAULTS: Form = { warmup: '30', bankroll: '1000', kelly: '0,25', minEv: '2' }

function build(f: Form): { req: BacktestRequest | null; errors: Errors } {
  const errors: Errors = {}
  const warmup = parseNum(f.warmup), bankroll = parseNum(f.bankroll), kelly = parseNum(f.kelly), minEv = parseNum(f.minEv)
  if (warmup === null || !Number.isInteger(warmup) || warmup < 5) errors.warmup = 'Entero a partir de 5.'
  if (bankroll === null || bankroll <= 0) errors.bankroll = 'Mayor que 0.'
  if (kelly === null || kelly <= 0 || kelly > 1) errors.kelly = 'Entre 0,01 y 1.'
  if (minEv === null || minEv < 0 || minEv > 100) errors.minEv = 'Entre 0 y 100.'
  if (Object.keys(errors).length > 0) return { req: null, errors }
  return { errors, req: { warmup: warmup!, bankroll: bankroll!, kelly_multiplier: kelly!, min_ev: minEv! / 100 } }
}

const OUTCOME = { H: 'Local', D: 'Empate', A: 'Visitante' } as const

export function Backtest({ runner, active }: { runner: Runner<BacktestRequest, BacktestResult>; active: boolean }) {
  const [form, setForm] = useState<Form>(DEFAULTS)
  const { req, errors } = useMemo(() => build(form), [form])
  const set = (k: keyof Form) => (v: string) => setForm((f) => ({ ...f, [k]: v }))

  // First visit to the tab runs the default backtest; later runs are the user's.
  const { run, status } = runner
  useEffect(() => {
    if (active && status === 'idle') run(build(DEFAULTS).req!)
  }, [active, status, run])

  const launch = (e?: FormEvent) => {
    e?.preventDefault()
    if (req) run(req)
  }

  const d = runner.data
  const state: RegionState = d ? (d.n_events === 0 || d.predictions.length === 0 ? 'empty' : 'ready') : runner.status === 'error' ? 'error' : 'loading'

  return (
    <div className="workspace">
      <form className="panel controls" onSubmit={launch} aria-label="Parámetros del backtesting">
        <fieldset>
          <legend>Backtesting walk-forward</legend>
          <div className="row2">
            <NumberField label="Calentamiento (partidos)" value={form.warmup} onChange={set('warmup')} error={errors.warmup} />
            <NumberField label="Bankroll inicial" value={form.bankroll} onChange={set('bankroll')} error={errors.bankroll} />
            <NumberField label="Fracción de Kelly" value={form.kelly} onChange={set('kelly')} error={errors.kelly} />
            <NumberField label="EV mínimo, %" value={form.minEv} onChange={set('minEv')} error={errors.minEv} />
          </div>
          <p className="note">Cada partido se predice solo con lo ocurrido antes. Se apuesta cuando el EV supera el mínimo.</p>
        </fieldset>
        <Button type="submit" icon={<Play size={16} aria-hidden="true" />} busy={runner.status === 'loading'} disabled={!req}>Ejecutar backtest</Button>
      </form>

      <div className="results">
        <Region
          state={state}
          error={runner.error}
          onRetry={launch}
          busy={runner.status === 'loading'}
          skeleton={<div className="stack"><Skeleton shape="figures" /><Skeleton shape="chart" /><Skeleton shape="table" /></div>}
          empty={
            <EmptyState title="Sin partidos que predecir" action={<Button variant="secondary" onClick={() => setForm(DEFAULTS)}>Restablecer parámetros</Button>}>
              Con este calentamiento no quedan partidos en el historial para evaluar. Reduce el número de partidos de calentamiento.
            </EmptyState>
          }
        >
          {d ? <BacktestResults result={d} bankroll={runner.req?.bankroll ?? 1000} /> : null}
        </Region>
      </div>
    </div>
  )
}

function BacktestResults({ result: r, bankroll }: { result: BacktestResult; bankroll: number }) {
  const last = r.predictions.slice(-20).reverse()
  return (
    <div className="stack">
      <header className="results__head">
        <h2>Resultados del backtest</h2>
        <p className="muted num">{fmtInt(r.n_events)} partidos evaluados tras {fmtInt(r.warmup)} de calentamiento.</p>
      </header>

      <dl className="stats stats--grid">
        <StatFigure label="Brier" value={fmtNum(r.brier_score, 4)} />
        <StatFigure label="Log-loss" value={fmtNum(r.log_loss, 4)} />
        <StatFigure label="Precisión" value={fmtPct(r.accuracy, 1)} />
        <StatFigure label="ROI" value={fmtSignedPct(r.roi, 1)} />
        <StatFigure label="Yield" value={`${fmtSigned(r.yield_pct, 1)} %`} />
        <StatFigure label="Beneficio" value={fmtSigned(r.profit, 2)} />
        <StatFigure label="Max drawdown" value={fmtPct(r.max_drawdown, 1)} />
        <StatFigure label="Sharpe" value={fmtNum(r.sharpe, 2)} />
        <StatFigure label="Nº apuestas" value={fmtInt(r.n_bets)} />
      </dl>

      <section className="panel" aria-labelledby="bt-curve">
        <h3 id="bt-curve">Curva de bankroll</h3>
        {r.bankroll_curve.length > 1
          ? <LineChart values={r.bankroll_curve} base={bankroll} unit="Apuesta" label="Curva de bankroll por apuesta" />
          : <p className="note">No hubo apuestas con el EV mínimo elegido: la curva se queda en el bankroll inicial. Baja el EV mínimo para ver apuestas.</p>}
      </section>

      <section className="panel" aria-labelledby="bt-cal">
        <h3 id="bt-cal">Calibración</h3>
        <CalibrationChart bins={r.calibration} />
      </section>

      <section className="panel" aria-labelledby="bt-last">
        <h3 id="bt-last">Últimas 20 predicciones</h3>
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Últimas veinte predicciones del backtest">
          <table>
            <thead>
              <tr>
                <th scope="col">Fecha</th>
                <th scope="col">Partido</th>
                <th scope="col" className="r">Marcador</th>
                <th scope="col" className="r">Local</th>
                <th scope="col" className="r">Empate</th>
                <th scope="col" className="r">Visitante</th>
                <th scope="col">Resultado</th>
                <th scope="col">Acierto</th>
              </tr>
            </thead>
            <tbody>
              {last.map((p) => {
                const pick = p.p_home >= p.p_draw && p.p_home >= p.p_away ? 'H' : p.p_draw >= p.p_away ? 'D' : 'A'
                const hit = pick === p.outcome
                return (
                  <tr key={`${p.date}-${p.home}-${p.away}`}>
                    <td className="num nowrap">{fmtDate(p.date)}</td>
                    <th scope="row">{p.home} <span className="muted">contra</span> {p.away}</th>
                    <td className="r num">{p.home_goals}-{p.away_goals}</td>
                    <td className="r num">{fmtPct(p.p_home, 1)}</td>
                    <td className="r num">{fmtPct(p.p_draw, 1)}</td>
                    <td className="r num">{fmtPct(p.p_away, 1)}</td>
                    <td>{OUTCOME[p.outcome]}</td>
                    <td>{hit ? <span className="hit"><Check size={16} aria-hidden="true" />Sí</span> : <span className="muted">No</span>}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
