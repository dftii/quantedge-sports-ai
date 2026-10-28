import { useMemo, useRef, useState, type KeyboardEvent } from 'react'
import type { FootballSimulationResult } from '../engine/types'
import { fmtNum, fmtPct } from './format'

interface Cell { h: number; a: number }

/** Eight quantised steps of the glyph ramp; sqrt spreads the long tail of low-probability scores. */
const level = (p: number, max: number): number => Math.min(7, Math.floor(Math.sqrt(p / max) * 8))

export function ScoreHeatmap({ result }: { result: FootballSimulationResult }) {
  const { matrix, home, away } = result.prediction
  const limit = matrix.length - 1
  const sizes = [4, 6, 8, 10].filter((s) => s <= limit)
  const [size, setSize] = useState(Math.min(6, limit))
  const [picked, setPicked] = useState<Cell | null>(null)
  const refs = useRef(new Map<string, HTMLButtonElement>())

  const { max, best } = useMemo(() => {
    let mx = 0
    let b: Cell = { h: 0, a: 0 }
    for (let h = 0; h <= size; h++) for (let a = 0; a <= size; a++) {
      const p = matrix[h]?.[a] ?? 0
      if (p > mx) { mx = p; b = { h, a } }
    }
    return { max: mx || 1, best: b }
  }, [matrix, size])

  const cur = picked && picked.h <= size && picked.a <= size ? picked : best
  const p = matrix[cur.h]?.[cur.a] ?? 0
  const rowTotal = (matrix[cur.h] ?? []).reduce((s, v) => s + v, 0)
  const colTotal = matrix.reduce((s, r) => s + (r[cur.a] ?? 0), 0)
  const outcome = cur.h > cur.a ? `Victoria de ${home}` : cur.h < cur.a ? `Victoria de ${away}` : 'Empate'
  const top = [...result.prediction.markets.correct_scores].sort((x, y) => y.probability - x.probability).slice(0, 5)

  const onKey = (e: KeyboardEvent, c: Cell) => {
    const d = e.key === 'ArrowRight' ? [0, 1] : e.key === 'ArrowLeft' ? [0, -1] : e.key === 'ArrowDown' ? [1, 0] : e.key === 'ArrowUp' ? [-1, 0] : null
    if (!d) return
    e.preventDefault()
    const n = { h: Math.min(size, Math.max(0, c.h + d[0])), a: Math.min(size, Math.max(0, c.a + d[1])) }
    refs.current.get(`${n.h}-${n.a}`)?.focus()
  }

  const idx = Array.from({ length: size + 1 }, (_, i) => i)

  return (
    <div className="workspace workspace--wide">
      <section className="panel" aria-labelledby="heat-title">
        <header className="panel__head">
          <h2 id="heat-title">Mapa de marcadores</h2>
          <div className="seg" role="radiogroup" aria-label="Goles máximos por equipo">
            {sizes.map((s) => (
              <label key={s}>
                <input type="radio" name="heat-size" value={s} checked={size === s} onChange={() => setSize(s)} />
                <span>Hasta {s}</span>
              </label>
            ))}
          </div>
        </header>
        <p className="note">Filas: goles de {home}. Columnas: goles de {away}. La diagonal marca los empates.</p>

        <div className="table-wrap" tabIndex={0} role="region" aria-label="Matriz de probabilidad de marcadores">
          <table className="heat">
            <thead>
              <tr>
                <th scope="col"><span className="visually-hidden">Local contra visitante</span></th>
                {idx.map((a) => <th key={a} scope="col" className={`num${cur.a === a ? ' is-on' : ''}`}>{a}</th>)}
              </tr>
            </thead>
            <tbody>
              {idx.map((h) => (
                <tr key={h}>
                  <th scope="row" className={`num${cur.h === h ? ' is-on' : ''}`}>{h}</th>
                  {idx.map((a) => {
                    const v = matrix[h]?.[a] ?? 0
                    const l = level(v, max)
                    const on = cur.h === h && cur.a === a
                    return (
                      <td key={a}>
                        <button
                          type="button"
                          ref={(el) => { if (el) refs.current.set(`${h}-${a}`, el); else refs.current.delete(`${h}-${a}`) }}
                          className={`cell cell--${l}${h === a ? ' cell--diag' : ''}${on ? ' is-on' : ''} num`}
                          tabIndex={on ? 0 : -1}
                          aria-label={`${home} ${h}, ${away} ${a}: ${fmtPct(v, 2)}`}
                          onPointerEnter={() => setPicked({ h, a })}
                          onFocus={() => setPicked({ h, a })}
                          onKeyDown={(e) => onKey(e, { h, a })}
                        >
                          {fmtNum(v * 100, 1)}
                        </button>
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="ramp" aria-hidden="true">
          <span className="muted">0 %</span>
          {Array.from({ length: 8 }, (_, i) => <span key={i} className={`ramp__step cell--${i}`} />)}
          <span className="muted num">{fmtPct(max, 1)}</span>
        </div>
      </section>

      <aside className="panel readout" aria-label="Detalle del marcador">
        <output aria-live="polite">
          <span className="readout__score figure num">{cur.h}-{cur.a}</span>
          <span className="readout__prob figure num">{fmtPct(p, 2)}</span>
          <span className="readout__line">{outcome}</span>
        </output>
        <dl className="kv">
          <div><dt>Cuota justa</dt><dd className="num">{p > 0 ? fmtNum(1 / p) : '—'}</dd></div>
          <div><dt>{home} marca {cur.h}</dt><dd className="num">{fmtPct(rowTotal, 2)}</dd></div>
          <div><dt>{away} marca {cur.a}</dt><dd className="num">{fmtPct(colTotal, 2)}</dd></div>
        </dl>
        <h3>Marcadores más probables</h3>
        <ol className="rank">
          {top.map((c) => (
            <li key={c.score}>
              <span className="num">{c.home_goals}-{c.away_goals}</span>
              <span className="num">{fmtPct(c.probability, 2)}</span>
            </li>
          ))}
        </ol>
      </aside>
    </div>
  )
}
