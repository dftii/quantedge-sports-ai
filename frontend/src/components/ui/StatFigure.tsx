import type { ReactNode } from 'react'

/** A figure and its label, as a tool would print them. Render inside <dl className="stats">. */
export function StatFigure({ label, value, note }: { label: string; value: ReactNode; note?: ReactNode }) {
  return (
    <div className="stat">
      <dt>{label}</dt>
      <dd className="figure num">{value}</dd>
      {note ? <dd className="stat__note">{note}</dd> : null}
    </div>
  )
}
