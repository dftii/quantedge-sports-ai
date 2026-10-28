import { useState } from 'react'
import type { CalibrationBin } from '../../engine/types'
import { fmtInt, fmtPct } from '../format'
import { useWidth } from './useWidth'

const M = { l: 48, r: 12, t: 12, b: 40 }

/** Predicted probability against observed frequency per bin; point area follows the bin count. */
export function CalibrationChart({ bins }: { bins: CalibrationBin[] }) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [active, setActive] = useState<number | null>(null)
  const pts = bins
    .map((b, i) => ({ b, i }))
    .filter(({ b }) => b.count > 0 && b.mean_predicted !== null && b.observed_frequency !== null)
  const maxCount = Math.max(1, ...pts.map(({ b }) => b.count))
  const side = Math.min(width, 440)
  const H = Math.round(side * 0.82)
  const plotW = side - M.l - M.r
  const plotH = H - M.t - M.b
  const x = (p: number) => M.l + p * plotW
  const y = (p: number) => M.t + plotH - p * plotH
  const grid = [0, 0.25, 0.5, 0.75, 1]
  const sel = pts.find(({ i }) => i === active)
  const line = pts.map(({ b }, k) => `${k === 0 ? 'M' : 'L'}${x(b.mean_predicted!).toFixed(1)} ${y(b.observed_frequency!).toFixed(1)}`).join('')

  return (
    <div className="chart-block">
      <p className="chart-readout num" aria-live="polite">
        {sel
          ? <>Predicho <b>{fmtPct(sel.b.mean_predicted!, 1)}</b>, observado <b>{fmtPct(sel.b.observed_frequency!, 1)}</b>, {fmtInt(sel.b.count)} casos</>
          : 'Pasa el cursor o enfoca un punto para ver su tramo.'}
      </p>
      <div ref={ref} className="chart">
        <svg width={side} height={H} viewBox={`0 0 ${side} ${H}`} role="group" aria-label="Curva de calibración: probabilidad predicha frente a frecuencia observada">
          {grid.map((g) => (
            <g key={g}>
              <line className="grid-line" x1={M.l} x2={M.l + plotW} y1={y(g)} y2={y(g)} />
              <text className="axis-text num" x={M.l - 8} y={y(g) + 4} textAnchor="end">{fmtPct(g, 0)}</text>
              <text className="axis-text num" x={x(g)} y={M.t + plotH + 16} textAnchor={g === 0 ? 'start' : g === 1 ? 'end' : 'middle'}>{fmtPct(g, 0)}</text>
            </g>
          ))}
          <line className="ref-line" x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} />
          <text className="axis-text" x={M.l + plotW / 2} y={H - 4} textAnchor="middle">Probabilidad predicha</text>
          <path className="series-line series-line--thin" d={line} />
          {pts.map(({ b, i }) => (
            <circle key={i} className={`point point--bin${active === i ? ' is-active' : ''}`} cx={x(b.mean_predicted!)} cy={y(b.observed_frequency!)}
              r={3 + 9 * Math.sqrt(b.count / maxCount)} tabIndex={0} role="img"
              aria-label={`Predicho ${fmtPct(b.mean_predicted!, 1)}, observado ${fmtPct(b.observed_frequency!, 1)}, ${b.count} casos`}
              onPointerEnter={() => setActive(i)} onPointerLeave={() => setActive(null)} onFocus={() => setActive(i)} onBlur={() => setActive(null)} />
          ))}
        </svg>
      </div>
      <p className="chart-note">Eje vertical: frecuencia observada. La diagonal es la calibración perfecta; el área del punto crece con el número de casos.</p>
    </div>
  )
}
