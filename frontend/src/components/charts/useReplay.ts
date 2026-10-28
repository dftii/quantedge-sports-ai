import { useEffect, useMemo, useState, useSyncExternalStore } from 'react'

export const STEPS = 28
const REDUCED = '(prefers-reduced-motion: reduce)'

export function useReducedMotion(): boolean {
  return useSyncExternalStore(
    (cb) => {
      const m = window.matchMedia(REDUCED)
      m.addEventListener('change', cb)
      return () => m.removeEventListener('change', cb)
    },
    () => window.matchMedia(REDUCED).matches,
    () => false,
  )
}

/** Engines may send counts (goal_histogram) or proportions of n (score_distribution): normalise to integer counts. */
export function toCounts(values: number[], n: number): number[] {
  const sum = values.reduce((a, b) => a + b, 0)
  return sum <= 1.5 ? values.map((v) => Math.round(v * n)) : values.map((v) => Math.round(v))
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function sampleOrder(counts: number[], seed: number): Uint8Array {
  const total = counts.reduce((a, b) => a + b, 0)
  const out = new Uint8Array(total)
  let k = 0
  counts.forEach((c, i) => { out.fill(i, k, k + c); k += c })
  const rnd = mulberry32(seed)
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1))
    const t = out[i]; out[i] = out[j]; out[j] = t
  }
  return out
}

function prefixCounts(sample: Uint8Array, bins: number, upto: number): number[] {
  const c = new Array<number>(bins).fill(0)
  for (let i = 0; i < upto; i++) c[sample[i]]++
  return c
}

/**
 * Replays a finished run: the final counts are expanded into a shuffled sample and the bars show the
 * counts over its first k draws, so they fill from 0 to the exact final totals in STEPS stepped frames.
 */
export function useReplay(series: number[][], runKey: number, duration = 700) {
  const reduced = useReducedMotion()
  const [round, setRound] = useState(0)
  const key = `${runKey}:${round}`
  const [anim, setAnim] = useState({ key: '', step: 0 })
  const samples = useMemo(
    () => series.map((c, i) => sampleOrder(c, runKey * 7919 + i * 104729 + round * 31)),
    [series, runKey, round],
  )

  useEffect(() => {
    if (reduced) return
    const t0 = performance.now()
    let raf = 0
    const tick = (now: number) => {
      const step = Math.min(STEPS, Math.floor(((now - t0) / duration) * STEPS))
      setAnim((a) => (a.key === key && a.step === step ? a : { key, step }))
      if (step < STEPS) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [key, reduced, duration])

  const step = reduced ? STEPS : anim.key === key ? anim.step : 0
  const counts = useMemo(
    () => samples.map((s, i) => prefixCounts(s, series[i].length, Math.round((s.length * step) / STEPS))),
    [samples, series, step],
  )
  return { counts, step, fraction: step / STEPS, done: step >= STEPS, replay: () => setRound((r) => r + 1) }
}
