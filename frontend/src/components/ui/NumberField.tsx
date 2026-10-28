import { useId } from 'react'

interface Props {
  label: string
  value: string
  onChange: (v: string) => void
  error?: string
  helper?: string
  placeholder?: string
  name?: string
}

// type="text" + inputMode="decimal" so Spanish users can type "1,85"; parsing lives in format.ts.
export function NumberField({ label, value, onChange, error, helper, placeholder, name }: Props) {
  const id = useId()
  const msg = error ?? helper
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <input id={id} name={name} type="text" inputMode="decimal" autoComplete="off" className="num" value={value} placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)} aria-invalid={error ? true : undefined} aria-describedby={msg ? `${id}-msg` : undefined} />
      <p className={`field__msg${error ? ' field__msg--error' : ''}`} id={`${id}-msg`}>{msg}</p>
    </div>
  )
}
