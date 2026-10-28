import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { Loader } from 'lucide-react'

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary'
  busy?: boolean
  icon?: ReactNode
}

export function Button({ variant = 'primary', busy = false, icon, children, className, type = 'button', ...rest }: Props) {
  return (
    <button type={type} className={`btn btn--${variant}${className ? ` ${className}` : ''}`} aria-busy={busy || undefined} {...rest}>
      {busy ? <Loader className="spin" size={16} aria-hidden="true" /> : icon}
      <span>{children}</span>
    </button>
  )
}
