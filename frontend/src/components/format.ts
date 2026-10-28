// Spanish number formatting. useGrouping 'always' keeps "4.200" (es-ES drops the dot under 5 digits by default).
const cache = new Map<string, Intl.NumberFormat>()
function nf(min: number, max: number): Intl.NumberFormat {
  const key = `${min}:${max}`
  let f = cache.get(key)
  if (!f) {
    f = new Intl.NumberFormat('es-ES', { minimumFractionDigits: min, maximumFractionDigits: max, useGrouping: 'always' } as Intl.NumberFormatOptions)
    cache.set(key, f)
  }
  return f
}

export const fmtInt = (n: number): string => nf(0, 0).format(Math.round(n))
export const fmtNum = (n: number, d = 2): string => nf(d, d).format(n)
export const fmtPct = (p: number, d = 1): string => `${nf(d, d).format(p * 100)}\u00a0%`
export const fmtSigned = (n: number, d = 2): string => `${n > 0 ? '+' : n < 0 ? '\u2212' : ''}${nf(d, d).format(Math.abs(n))}`
export const fmtSignedPct = (p: number, d = 1): string => `${fmtSigned(p * 100, d)}\u00a0%`
export const fmtDate = (iso: string): string =>
  new Date(`${iso}T00:00:00`).toLocaleDateString('es-ES', { day: '2-digit', month: 'short', year: 'numeric' })

/** Accepts "1,85" and "1.85"; empty or non-numeric gives null. */
export function parseNum(s: string): number | null {
  const t = s.trim().replace(',', '.')
  if (t === '') return null
  const v = Number(t)
  return Number.isFinite(v) ? v : null
}

export const humanize = (key: string): string => {
  const t = key.replace(/_/g, ' ')
  return t.charAt(0).toUpperCase() + t.slice(1)
}
