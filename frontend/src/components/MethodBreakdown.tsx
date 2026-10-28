import type { FightSimulationResult, MethodProbs } from '../engine/types'
import { BarList } from './charts/BarList'
import { fmtNum, fmtPct, fmtSigned, humanize } from './format'

const METHODS: { key: keyof MethodProbs; label: string }[] = [
  { key: 'ko_tko', label: 'KO/TKO' },
  { key: 'submission', label: 'Sumisión' },
  { key: 'decision', label: 'Decisión' },
]

export function MethodBreakdown({ result }: { result: FightSimulationResult }) {
  const { prediction: p, simulation: sim } = result
  const all = [...METHODS.flatMap((m) => [p.method_probs.a[m.key], p.method_probs.b[m.key]])]
  const maxM = Math.max(...all, 1e-9)

  const roundKeys = [...Object.keys(p.finish_round_probs).filter((k) => k !== 'decision').sort((a, b) => Number(a) - Number(b)), ...('decision' in p.finish_round_probs ? ['decision'] : [])]
  const maxR = Math.max(...roundKeys.map((k) => p.finish_round_probs[k]), 1e-9)

  const factors = Object.entries(p.factors)
  const maxF = Math.max(...factors.map(([, v]) => Math.abs(v)), 1e-9)
  const logit = factors.reduce((s, [, v]) => s + v, 0)

  return (
    <div className="stack">
      <header className="results__head">
        <h2>{p.fighter_a} contra {p.fighter_b}</h2>
        <p className="muted">Cómo se resuelve el combate según el modelo, método por método.</p>
      </header>

      <section className="panel" aria-labelledby="mb-methods">
        <h3 id="mb-methods">Método de victoria</h3>
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Probabilidad de cada método por luchador">
          <table>
            <thead>
              <tr>
                <th scope="col">Método</th>
                <th scope="col" className="r">{p.fighter_a}, analítica</th>
                <th scope="col" className="r">{p.fighter_a}, Monte Carlo</th>
                <th scope="col" className="r">{p.fighter_b}, analítica</th>
                <th scope="col" className="r">{p.fighter_b}, Monte Carlo</th>
              </tr>
            </thead>
            <tbody>
              {METHODS.map((m) => (
                <tr key={m.key}>
                  <th scope="row">{m.label}</th>
                  <td className="r num">{fmtPct(p.method_probs.a[m.key], 1)}</td>
                  <td className="r num">{fmtPct(sim.methods.a[m.key], 1)}</td>
                  <td className="r num">{fmtPct(p.method_probs.b[m.key], 1)}</td>
                  <td className="r num">{fmtPct(sim.methods.b[m.key], 1)}</td>
                </tr>
              ))}
              <tr className="total">
                <th scope="row">Victoria total</th>
                <td className="r num">{fmtPct(p.win_prob_a, 1)}</td>
                <td className="r num">{fmtPct(sim.win_prob_a, 1)}</td>
                <td className="r num">{fmtPct(p.win_prob_b, 1)}</td>
                <td className="r num">{fmtPct(sim.win_prob_b, 1)}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="cols2">
          {([['a', p.fighter_a], ['b', p.fighter_b]] as const).map(([side, name]) => (
            <div key={side}>
              <h4>{name}</h4>
              <BarList
                label={`Métodos de victoria de ${name}`}
                items={METHODS.map((m) => ({ key: m.key, label: m.label, ratio: p.method_probs[side][m.key] / maxM, text: fmtPct(p.method_probs[side][m.key], 1), tone: side === 'a' ? 'accent' : 'muted' }))}
              />
            </div>
          ))}
        </div>
      </section>

      <section className="panel" aria-labelledby="mb-rounds">
        <h3 id="mb-rounds">Final por asalto</h3>
        <BarList
          label="Probabilidad de que el combate termine en cada asalto o llegue a decisión"
          items={roundKeys.map((k) => ({ key: k, label: k === 'decision' ? 'Decisión' : `Asalto ${k}`, ratio: p.finish_round_probs[k] / maxR, text: fmtPct(p.finish_round_probs[k], 1) }))}
        />
        <p className="note num">Duración esperada: {fmtNum(p.expected_rounds)} asaltos.</p>
      </section>

      <section className="panel" aria-labelledby="mb-factors">
        <h3 id="mb-factors">Factores del modelo, en logit</h3>
        <p className="note">Positivo favorece a {p.fighter_a}; negativo, a {p.fighter_b}. La suma de los factores es el logit de la victoria de {p.fighter_a}.</p>
        <BarList
          diverging
          label={`Contribución de cada factor al logit de victoria de ${p.fighter_a}`}
          items={factors.map(([k, v]) => ({ key: k, label: humanize(k), ratio: v / maxF, text: fmtSigned(v, 2) }))}
        />
        <p className="note num">Suma: {fmtSigned(logit, 2)}, equivale a {fmtPct(p.win_prob_a, 1)} para {p.fighter_a}.</p>
      </section>
    </div>
  )
}
