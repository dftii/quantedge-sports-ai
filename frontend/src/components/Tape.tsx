import type { ReactNode } from 'react'

export interface TapeRow {
  label: string
  left: ReactNode
  right: ReactNode
  /** which side holds the better value; the other side is set in muted ink */
  edge?: 'left' | 'right' | null
}

/** Side-by-side comparison of two competitors: left value, metric, right value. */
export function Tape({ leftName, rightName, rows, caption }: { leftName: string; rightName: string; rows: TapeRow[]; caption: string }) {
  return (
    <div className="table-wrap" tabIndex={0} role="region" aria-label={caption}>
      <table className="tape">
        <caption className="visually-hidden">{caption}</caption>
        <thead>
          <tr>
            <th scope="col" className="tape__l">{leftName}</th>
            <th scope="col"><span className="visually-hidden">Métrica</span></th>
            <th scope="col" className="tape__r">{rightName}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label}>
              <td className={`tape__l num${r.edge === 'right' ? ' is-worse' : ''}`}>{r.left}</td>
              <th scope="row" className="tape__m">{r.label}</th>
              <td className={`tape__r num${r.edge === 'left' ? ' is-worse' : ''}`}>{r.right}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Last results as W/D/L glyphs, newest last. */
export function FormGlyphs({ form }: { form: string }) {
  const names: Record<string, string> = { W: 'victoria', D: 'empate', L: 'derrota' }
  return (
    <span className="form" role="img" aria-label={form.split('').map((c) => names[c] ?? c).join(', ')}>
      {form.split('').map((c, i) => <span key={i} className={`form__g form__g--${c}`} aria-hidden="true">{c}</span>)}
    </span>
  )
}

/** Higher or lower wins: returns the better side or null on a tie. */
export function edge(a: number, b: number, better: 'high' | 'low'): 'left' | 'right' | null {
  if (a === b) return null
  return (a > b) === (better === 'high') ? 'left' : 'right'
}
