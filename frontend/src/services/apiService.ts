// QuantService backed by the FastAPI server (backend/app/main.py).
import type {
  BacktestRequest, BacktestResult, Dataset, FightSimulationRequest, FightSimulationResult, FootballSimulationRequest,
  FootballSimulationResult, QuantService, SportEvent,
} from '../engine/types'

const EVENTS_PAGE = 500 // the API's maximum page size

/** FastAPI `detail` is a string, or a list of validation errors ({loc, msg}). */
function detailMessage(detail: unknown, status: number): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((d: { loc?: unknown[]; msg?: string }) => `${(d.loc ?? []).filter((p) => p !== 'body').join('.')}: ${d.msg}`).join('; ')
  }
  return `Error HTTP ${status}`
}

export function createApiService(baseUrl: string): QuantService {
  const root = baseUrl.replace(/\/+$/, '')

  async function request<T>(path: string, body?: unknown): Promise<T> {
    let res: Response
    try {
      res = await fetch(root + path, body === undefined ? undefined : {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      })
    } catch {
      throw new Error(`No se pudo conectar con la API en ${root}`)
    }
    if (!res.ok) {
      const payload = await res.json().catch(() => null)
      throw new Error(detailMessage(payload?.detail, res.status))
    }
    return res.json() as Promise<T>
  }

  async function allEvents(): Promise<SportEvent[]> {
    const events: SportEvent[] = []
    for (let offset = 0; ; offset += EVENTS_PAGE) {
      const page = await request<SportEvent[]>(`/api/events?limit=${EVENTS_PAGE}&offset=${offset}`)
      events.push(...page)
      if (page.length < EVENTS_PAGE) return events
    }
  }

  return {
    mode: 'api',
    async loadDataset(): Promise<Dataset> {
      const [competitions, teams, fighters, events] = await Promise.all([
        request<Dataset['competitions']>('/api/competitions'),
        request<Dataset['teams']>('/api/teams'),
        request<Dataset['fighters']>('/api/fighters'),
        allEvents(),
      ])
      return { competitions, teams, fighters, events }
    },
    simulateFootball: (req: FootballSimulationRequest) => request<FootballSimulationResult>('/api/simulate/football', { ...req, save: false }),
    simulateFight: (req: FightSimulationRequest) => request<FightSimulationResult>('/api/simulate/fight', { ...req, save: false }),
    backtestFootball: (req: BacktestRequest) => request<BacktestResult>('/api/backtest/football', req),
  }
}
