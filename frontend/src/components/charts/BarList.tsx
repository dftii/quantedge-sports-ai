import type { ReactNode } from 'react'

export interface BarItem {
  key: string
  label: ReactNode
  /** 0..1 of the track; -1..0 grows leftwards in a diverging list */
  ratio: number
  text: string
  tone?: 'accent' | 'muted'
}

/** Horizontal bars drawn with transform: scaleX only (never width), from a zero baseline. */
export function BarList({ items, diverging = false, label }: { items: BarItem[]; diverging?: boolean; label: string }) {
  return (
    <ul className={`bars${diverging ? ' bars--div' : ''}`} aria-label={label}>
      {items.map((it) => {
        const tone = it.tone ?? (it.ratio < 0 ? 'muted' : 'accent')
        return (
          <li key={it.key}>
            <span className="bars__label">{it.label}</span>
            <span className="bars__track" aria-hidden="true">
              <span className={`bars__fill bars__fill--${tone}${it.ratio < 0 ? ' bars__fill--neg' : ''}`} style={{ transform: `scaleX(${Math.min(1, Math.abs(it.ratio))})` }} />
            </span>
            <span className="bars__value num">{it.text}</span>
          </li>
        )
      })}
    </ul>
  )
}
