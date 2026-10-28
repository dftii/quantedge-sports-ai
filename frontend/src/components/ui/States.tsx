import type { ReactNode } from 'react'
import { CircleAlert, Inbox, RotateCcw } from 'lucide-react'
import { Button } from './Button'

type Shape = 'table' | 'chart' | 'figures' | 'grid' | 'bars'

/** Placeholder in the shape of what is coming. */
export function Skeleton({ shape, label = 'Cargando' }: { shape: Shape; label?: string }) {
  return (
    <div className={`skeleton skeleton--${shape}`} role="status" aria-label={label}>
      {shape === 'figures' && Array.from({ length: 4 }, (_, i) => <span key={i} className="skeleton__block" />)}
      {shape === 'table' && Array.from({ length: 6 }, (_, i) => <span key={i} className="skeleton__line" />)}
      {shape === 'chart' && Array.from({ length: 8 }, (_, i) => <span key={i} className="skeleton__bar" style={{ height: `${28 + ((i * 37) % 60)}%` }} />)}
      {shape === 'bars' && Array.from({ length: 5 }, (_, i) => <span key={i} className="skeleton__line" style={{ width: `${90 - i * 14}%` }} />)}
      {shape === 'grid' && Array.from({ length: 49 }, (_, i) => <span key={i} className="skeleton__cell" />)}
    </div>
  )
}

export function EmptyState({ title, children, action }: { title: string; children: ReactNode; action?: ReactNode }) {
  return (
    <div className="state state--empty">
      <Inbox size={20} aria-hidden="true" />
      <h3>{title}</h3>
      <p>{children}</p>
      {action}
    </div>
  )
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="state state--error" role="alert">
      <CircleAlert size={20} aria-hidden="true" />
      <h3>No se pudo completar la operación</h3>
      <p>{message}</p>
      {onRetry ? <Button variant="secondary" icon={<RotateCcw size={16} aria-hidden="true" />} onClick={onRetry}>Reintentar</Button> : null}
    </div>
  )
}

export type RegionState = 'loading' | 'error' | 'empty' | 'ready'

/** The four states of a data region: loading skeleton, empty, error with retry, full. */
export function Region({ state, error, onRetry, skeleton, empty, busy = false, children }: {
  state: RegionState
  error?: string | null
  onRetry?: () => void
  skeleton: ReactNode
  empty?: ReactNode
  busy?: boolean
  children: ReactNode
}) {
  if (state === 'error') return <ErrorState message={error ?? 'Error desconocido'} onRetry={onRetry} />
  if (state === 'loading') return <>{skeleton}</>
  if (state === 'empty') return <>{empty}</>
  return <div className="region" aria-busy={busy || undefined} data-stale={busy || undefined}>{children}</div>
}
