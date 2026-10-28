import { useState } from 'react'
import { Calculator } from 'lucide-react'
import type { ValueBet } from '../engine/types'
import { fmtNum, fmtPct, fmtSignedPct, parseNum } from './format'
import { Button } from './ui/Button'
import { NumberField } from './ui/NumberField'
import { EmptyState, Region, Skeleton, type RegionState } from './ui/States'
import { StatFigure } from './ui/StatFigure'

export interface ScanInput {
  status: 'idle' | 'loading' | 'ready' | 'error'
  error: string | null
  matchup: string | null
  valueBets: ValueBet[] | null
  margin: number | null
  bankroll: number
  kelly: number
}

const SELECTION: Record<string, string> = { home: 'Local', draw: 'Empate', away: 'Visitante' }
const selectionLabel = (s: string) => SELECTION[s.toLowerCase()] ?? s

export function ValueScanner({ scan, onGoSimulator }: { scan: ScanInput; onGoSimulator: () => void }) {
  const state: RegionState = scan.valueBets ? 'ready' : scan.status === 'error' ? 'error' : scan.status === 'ready' ? 'empty' : 'loading'
  const bets = scan.valueBets ?? []
  const best = bets.reduce<ValueBet | null>((b, v) => (!b || v.expected_value > b.expected_value ? v : b), null)

  return (
    <div className="workspace workspace--wide">
      <section className="panel" aria-labelledby="ev-title">
        <header className="panel__head">
          <h2 id="ev-title">Escáner de valor esperado</h2>
          {scan.matchup ? <p className="muted">{scan.matchup}</p> : null}
        </header>
        <Region
          state={state}
          error={scan.error}
          busy={scan.status === 'loading'}
          skeleton={<><Skeleton shape="figures" /><Skeleton shape="table" /></>}
          empty={
            <EmptyState title="Sin cuotas en la última simulación" action={<Button variant="secondary" onClick={onGoSimulator}>Ir al simulador</Button>}>
              El escáner compara la probabilidad del modelo con la del mercado. Introduce las cuotas del enfrentamiento y lanza la simulación.
            </EmptyState>
          }
        >
          <dl className="stats">
            <StatFigure label="Margen de la casa" value={scan.margin === null ? '—' : fmtPct(scan.margin, 2)} note="suma de probabilidades implícitas menos 1" />
            <StatFigure label="Selecciones con valor" value={`${bets.filter((b) => b.is_value).length} de ${bets.length}`} />
            <StatFigure label="Mejor EV" value={best ? fmtSignedPct(best.expected_value, 1) : '—'} note={best ? selectionLabel(best.selection) : undefined} />
          </dl>
          <div className="table-wrap" tabIndex={0} role="region" aria-label="Selecciones con cuota, probabilidad y valor esperado">
            <table className="ev">
              <thead>
                <tr>
                  <th scope="col">Selección</th>
                  <th scope="col" className="r">Cuota</th>
                  <th scope="col" className="r">Prob. implícita</th>
                  <th scope="col" className="r">Prob. justa</th>
                  <th scope="col" className="r">Prob. modelo</th>
                  <th scope="col" className="r">Ventaja</th>
                  <th scope="col" className="r">EV</th>
                  <th scope="col" className="r">Kelly</th>
                  <th scope="col" className="r">Stake</th>
                </tr>
              </thead>
              <tbody>
                {bets.map((b) => (
                  <tr key={b.selection} className={b.is_value ? 'is-value' : undefined}>
                    <th scope="row">{selectionLabel(b.selection)}{b.is_value ? <span className="tag">+EV</span> : null}</th>
                    <td className="r num">{fmtNum(b.decimal_odds)}</td>
                    <td className="r num">{fmtPct(b.implied_prob, 1)}</td>
                    <td className="r num">{fmtPct(b.fair_prob, 1)}</td>
                    <td className="r num">{fmtPct(b.model_prob, 1)}</td>
                    <td className="r num">{fmtSignedPct(b.edge, 1)}</td>
                    <td className={`r num${b.is_value ? '' : ' muted'}`}>{fmtSignedPct(b.expected_value, 1)}</td>
                    <td className="r num">{fmtPct(b.kelly_fraction, 2)}</td>
                    <td className="r num">{fmtNum(b.stake)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">Bankroll de la simulación: {fmtNum(scan.bankroll, 0)} unidades, Kelly al {fmtPct(scan.kelly, 0)}.</p>
        </Region>
      </section>

      <EvCalculator bankroll={scan.bankroll} kelly={scan.kelly} />
    </div>
  )
}

function EvCalculator({ bankroll, kelly }: { bankroll: number; kelly: number }) {
  const [prob, setProb] = useState('55')
  const [odds, setOdds] = useState('2,00')
  const [mult, setMult] = useState(fmtNum(kelly))
  const [cap, setCap] = useState('5')

  const p = parseNum(prob), o = parseNum(odds), m = parseNum(mult), c = parseNum(cap)
  const errors = {
    prob: p === null || p <= 0 || p >= 100 ? 'Entre 0 y 100.' : undefined,
    odds: o === null || o <= 1 ? 'Mayor que 1,00.' : undefined,
    mult: m === null || m <= 0 || m > 1 ? 'Entre 0,01 y 1.' : undefined,
    cap: c === null || c <= 0 || c > 100 ? 'Entre 0 y 100.' : undefined,
  }
  const ok = !errors.prob && !errors.odds && !errors.mult && !errors.cap
  // EV = p*o - 1; Kelly = multiplier * (p*o - 1) / (o - 1), floored at 0 and capped.
  const ev = ok ? (p! / 100) * o! - 1 : null
  const k = ok ? Math.min(c! / 100, Math.max(0, (m! * ((p! / 100) * o! - 1)) / (o! - 1))) : null

  return (
    <aside className="panel calc" aria-labelledby="calc-title">
      <h2 id="calc-title"><Calculator size={16} aria-hidden="true" /> Calculadora</h2>
      <div className="row2">
        <NumberField label="Probabilidad, %" value={prob} onChange={setProb} error={errors.prob} />
        <NumberField label="Cuota decimal" value={odds} onChange={setOdds} error={errors.odds} />
        <NumberField label="Fracción de Kelly" value={mult} onChange={setMult} error={errors.mult} />
        <NumberField label="Tope por apuesta, %" value={cap} onChange={setCap} error={errors.cap} />
      </div>
      <dl className="stats stats--stack" aria-live="polite">
        <StatFigure label="Valor esperado" value={ev === null ? '—' : fmtSignedPct(ev, 2)} note={ev === null ? undefined : ev > 0 ? 'por unidad apostada, con ventaja' : 'por unidad apostada, sin ventaja'} />
        <StatFigure label="Kelly" value={k === null ? '—' : fmtPct(k, 2)} note="del bankroll" />
        <StatFigure label="Stake" value={k === null ? '—' : fmtNum(k * bankroll)} note={`sobre ${fmtNum(bankroll, 0)} unidades`} />
      </dl>
    </aside>
  )
}
