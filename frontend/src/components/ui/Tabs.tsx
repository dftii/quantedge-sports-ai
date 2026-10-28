import { useRef, type KeyboardEvent, type ReactNode } from 'react'

export interface TabItem<T extends string> { id: T; label: string; icon?: ReactNode }

interface Props<T extends string> {
  items: TabItem<T>[]
  value: T
  onChange: (id: T) => void
  label: string
  idPrefix: string
  /** false when the tabs only switch a mode and no tabpanel carries their id */
  panels?: boolean
  className?: string
}

export function Tabs<T extends string>({ items, value, onChange, label, idPrefix, panels = true, className = '' }: Props<T>) {
  const refs = useRef<(HTMLButtonElement | null)[]>([])
  const onKey = (e: KeyboardEvent, i: number) => {
    const last = items.length - 1
    const next = e.key === 'ArrowRight' ? (i === last ? 0 : i + 1) : e.key === 'ArrowLeft' ? (i === 0 ? last : i - 1) : e.key === 'Home' ? 0 : e.key === 'End' ? last : -1
    if (next < 0) return
    e.preventDefault()
    onChange(items[next].id)
    refs.current[next]?.focus()
  }
  return (
    <div role="tablist" aria-label={label} className={`tabs ${className}`}>
      {items.map((it, i) => (
        <button
          key={it.id}
          ref={(el) => { refs.current[i] = el }}
          type="button"
          role="tab"
          id={`${idPrefix}-tab-${it.id}`}
          aria-selected={it.id === value}
          aria-controls={panels ? `${idPrefix}-panel-${it.id}` : undefined}
          tabIndex={it.id === value ? 0 : -1}
          className="tab"
          onClick={() => onChange(it.id)}
          onKeyDown={(e) => onKey(e, i)}
        >
          {it.icon}
          <span>{it.label}</span>
        </button>
      ))}
    </div>
  )
}
