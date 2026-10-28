import { useMemo } from 'react'
import { RotateCcw } from 'lucide-react'
import { Button } from '../ui/Button'
import { fmtInt, fmtPct } from '../format'
import { BarList } from './BarList'
import { useReplay } from './useReplay'
import { useWidth } from './useWidth'

export interface HistSeries { name: string; counts: number[]; tone: 'accent' | 'soft' }
export interface Ranked { title: string; labels: string[]; counts: number[] }

interface Props {
  title: string
  categories: string[]
  series: HistSeries[]
  ranked?: Ranked
  total: number
  runKey: number
}

const H = 220
const M = { l: 48, r: 8, t: 10, b: 28 }

function niceMax(share: number): number {
  const step = share >= 0.2 ? 0.1 : share >= 0.04 ? 0.05 : 0.01
  return Math.max(step, Math.ceil(share / step) * step)
}

/**
 * The Monte Carlo run: vertical bars (and optionally a ranked list) refill from zero to the final counts,
 * drawn with scaleY/scaleX only, beside a running "n = k / total" counter.
 */
export function RunDistribution({ title, categories, series, ranked, total, runKey }: Props) {
  const all = useMemo(() => [...series.map((s) => s.counts), ...(ranked ? [ranked.counts] : [])], [series, ranked])
  const rep = useReplay(all, runKey)
  const [ref, width] = useWidth<HTMLDivElement>()

  const plotW = width - M.l - M.r
  const plotH = H - M.t - M.b
  const slot = plotW / categories.length
  const barW = Math.min(24, (slot - 6) / series.length)
  const maxFinal = Math.max(1, ...series.flatMap((s) => s.counts))
  const yMaxShare = niceMax(maxFinal / total)
  const yMax = yMaxShare * total
  const ticks = [0, yMaxShare / 2, yMaxShare]
  const tickDecimals = (yMaxShare * 100) % 1 === 0 && ((yMaxShare / 2) * 100) % 1 === 0 ? 0 : 1
  const shown = Math.round(total * rep.fraction)
  const rankedMax = ranked ? Math.max(1, ...ranked.counts) : 1

  return (
    <section className="panel run" aria-labelledby="run-title">
      <header className="run__head">
        <h3 id="run-title">{title}</h3>
        <p className="run__counter num" aria-label={`${fmtInt(shown)} de ${fmtInt(total)} simulaciones`}>
          n = <b>{fmtInt(shown)}</b> / {fmtInt(total)}
        </p>
        <Button variant="secondary" className="btn--sm" icon={<RotateCcw size={16} aria-hidden="true" />} onClick={rep.replay}>Repetir</Button>
      </header>

      <ul className="legend">
        {series.map((s) => (
          <li key={s.name}><span className={`swatch swatch--${s.tone}`} aria-hidden="true" />{s.name}</li>
        ))}
      </ul>

      <div ref={ref} className="chart">
        <svg width={width} height={H} viewBox={`0 0 ${width} ${H}`} role="img" aria-label={`${title}: frecuencia de cada resultado sobre ${fmtInt(total)} simulaciones`}>
          {ticks.map((t) => {
            const y = M.t + plotH - (t / yMaxShare) * plotH
            return (
              <g key={t}>
                <line className="grid-line" x1={M.l} x2={width - M.r} y1={y} y2={y} />
                <text className="axis-text num" x={M.l - 8} y={y + 4} textAnchor="end">{fmtPct(t, tickDecimals)}</text>
              </g>
            )
          })}
          {categories.map((c, ci) => (
            <text key={c} className="axis-text num" x={M.l + slot * ci + slot / 2} y={H - 8} textAnchor="middle">{c}</text>
          ))}
          {series.map((s, si) =>
            categories.map((c, ci) => {
              const live = rep.counts[si]?.[ci] ?? 0
              const x = M.l + slot * ci + (slot - barW * series.length) / 2 + si * barW
              return (
                <rect key={`${si}-${c}`} className={`bar-v bar-v--${s.tone}`} x={x} y={M.t} width={barW - 1} height={plotH}
                  style={{ transform: `scaleY(${Math.min(1, live / yMax)})` }}>
                  <title>{`${s.name}, ${c}: ${fmtInt(live)} de ${fmtInt(total)}`}</title>
                </rect>
              )
            }),
          )}
        </svg>
      </div>

      {ranked ? (
        <div className="run__ranked">
          <h4>{ranked.title}</h4>
          <BarList
            label={ranked.title}
            items={ranked.labels.map((l, i) => ({
              key: l,
              label: l,
              ratio: (rep.counts[series.length]?.[i] ?? 0) / rankedMax,
              text: fmtPct((rep.counts[series.length]?.[i] ?? 0) / total, 2),
            }))}
          />
        </div>
      ) : null}
    </section>
  )
}
