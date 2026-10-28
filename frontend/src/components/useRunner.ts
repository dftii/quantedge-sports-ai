import { useCallback, useRef, useState } from 'react'

/** Async call with stale-result protection. `data`/`req` always hold the last successful run, so views stay on screen while a rerun is in flight. */
export interface Runner<Req, Res> {
  status: 'idle' | 'loading' | 'ready' | 'error'
  data: Res | null
  req: Req | null
  error: string | null
  runId: number
  run: (req: Req) => void
}

interface Inner<Req, Res> { status: Runner<Req, Res>['status']; data: Res | null; req: Req | null; error: string | null; runId: number }

export function useRunner<Req, Res>(fn: (req: Req) => Promise<Res>): Runner<Req, Res> {
  const [state, setState] = useState<Inner<Req, Res>>({ status: 'idle', data: null, req: null, error: null, runId: 0 })
  const seq = useRef(0)
  const fnRef = useRef(fn)
  fnRef.current = fn
  const run = useCallback((req: Req) => {
    const id = ++seq.current
    setState((s) => ({ ...s, status: 'loading', error: null }))
    fnRef.current(req).then(
      (data) => { if (id === seq.current) setState((s) => ({ status: 'ready', data, req, error: null, runId: s.runId + 1 })) },
      (e: unknown) => { if (id === seq.current) setState((s) => ({ ...s, status: 'error', error: e instanceof Error ? e.message : String(e) })) },
    )
  }, [])
  return { ...state, run }
}
