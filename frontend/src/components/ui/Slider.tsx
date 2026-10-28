import { useId } from 'react'

interface Props {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
  format: (v: number) => string
}

export function Slider({ label, value, min, max, step, onChange, format }: Props) {
  const id = useId()
  return (
    <div className="field slider">
      <div className="slider__head">
        <label htmlFor={id}>{label}</label>
        <output htmlFor={id} className="num">{format(value)}</output>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))}
        aria-valuetext={format(value)} />
    </div>
  )
}
