import { useState, type KeyboardEvent, type PointerEvent } from 'react'
import { fmtInt } from '../format'
import { useWidth } from './useWidth'

interface Props {
  values: number[]
  /** reference level drawn as a muted rule (the starting bankroll) */
  base: number
  label: string
  unit: string
  format?: (v: number) => string
}

const H = 240
const M = { l: 64, r: 16, t: 12, b: 28 }

export function LineChart({ values, base, label, unit, format = fmtInt }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<number | null>(null)
  const n = values.length
  const idx = hover ?? n - 1

  const lo = Math.min(base, ...values)
  const hi = Math.max(base, ...values)
  const pad = (hi - lo || 1) * 0.08
  const y0 = lo - pad
  const y1 = hi + pad
  const plotW = width - M.l - M.r
  const plotH = H - M.t - M.b
  const x = (i: number) => M.l + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2)
  const y = (v: number) => M.t + plotH - ((v - y0) / (y1 - y0)) * plotH
  const ticks = [0, 1, 2, 3].map((k) => y0 + ((y1 - y0) * k) / 3)
  const xTicks = n > 1 ? [0, Math.round((n - 1) / 2), n - 1] : [0]
  const path = values.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join('')

  const onMove = (e: PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const f = (e.clientX - r.left - M.l) / plotW
    setHover(Math.min(n - 1, Math.max(0, Math.round(f * (n - 1)))))
  }
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'ArrowLeft') setHover(Math.max(0, idx - 1))
    else if (e.key === 'ArrowRight') setHover(Math.min(n - 1, idx + 1))
    else if (e.key === 'Home') setHover(0)
    else if (e.key === 'End') setHover(n - 1)
    else return
    e.preventDefault()
  }

  return (
    <div className="chart-block">
      <p className="chart-readout num" aria-live="polite">
        {unit} {idx}: <b>{format(values[idx])}</b>
      </p>
      <div ref={ref} className="chart">
        <svg width={width} height={H} viewBox={`0 0 ${width} ${H}`} role="img" tabIndex={0} aria-label={`${label}. Flechas izquierda y derecha para recorrer la serie.`}
          onPointerMove={onMove} onPointerLeave={() => setHover(null)} onKeyDown={onKey} onBlur={() => setHover(null)}>
          {ticks.map((t) => (
            <g key={t}>
              <line className="grid-line" x1={M.l} x2={width - M.r} y1={y(t)} y2={y(t)} />
              <text className="axis-text num" x={M.l - 8} y={y(t) + 4} textAnchor="end">{format(t)}</text>
            </g>
          ))}
          <line className="ref-line" x1={M.l} x2={width - M.r} y1={y(base)} y2={y(base)} />
          {xTicks.map((i) => (
            <text key={i} className="axis-text num" x={x(i)} y={H - 8} textAnchor={i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle'}>{i}</text>
          ))}
          <path className="series-line" d={path} />
          <line className="cursor-line" x1={x(idx)} x2={x(idx)} y1={M.t} y2={M.t + plotH} />
          <circle className="point" cx={x(idx)} cy={y(values[idx])} r={3.5} />
        </svg>
      </div>
    </div>
  )
}
