import { useId } from 'react'
import { ChevronDown } from 'lucide-react'

export interface Option { value: string | number; label: string; disabled?: boolean }

interface Props {
  label: string
  value: string | number
  options: Option[]
  onChange: (v: string) => void
  error?: string
  helper?: string
}

export function Select({ label, value, options, onChange, error, helper }: Props) {
  const id = useId()
  const msg = error ?? helper
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <div className="select">
        <select id={id} value={value} onChange={(e) => onChange(e.target.value)} aria-invalid={error ? true : undefined} aria-describedby={msg ? `${id}-msg` : undefined}>
          {options.map((o) => <option key={o.value} value={o.value} disabled={o.disabled}>{o.label}</option>)}
        </select>
        <ChevronDown size={16} aria-hidden="true" />
      </div>
      <p className={`field__msg${error ? ' field__msg--error' : ''}`} id={`${id}-msg`}>{msg}</p>
    </div>
  )
}
