import { useCallback, useMemo, useSyncExternalStore } from 'react'

// App state lives in the URL hash (#s=football&t=sim&h=1&a=2) so a view can be shared and the back button works.
function subscribe(cb: () => void): () => void {
  for (const ev of ['hashchange', 'popstate', 'qe-hash']) window.addEventListener(ev, cb)
  return () => { for (const ev of ['hashchange', 'popstate', 'qe-hash']) window.removeEventListener(ev, cb) }
}

export type ParamPatch = Record<string, string | null>

export function useHashParams() {
  const hash = useSyncExternalStore(subscribe, () => window.location.hash, () => '')
  const params = useMemo(() => new URLSearchParams(hash.slice(1)), [hash])
  const update = useCallback((patch: ParamPatch, push = false) => {
    const next = new URLSearchParams(window.location.hash.slice(1))
    for (const [k, v] of Object.entries(patch)) {
      if (v === null) next.delete(k)
      else next.set(k, v)
    }
    const url = `#${next.toString()}`
    if (url === window.location.hash) return
    if (push) window.history.pushState(null, '', url)
    else window.history.replaceState(null, '', url)
    window.dispatchEvent(new Event('qe-hash'))
  }, [])
  return [params, update] as const
}
