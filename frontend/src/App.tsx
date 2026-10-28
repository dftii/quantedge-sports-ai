import { useCallback, useEffect, useState, type ReactNode } from 'react'
import type {
  BacktestRequest, BacktestResult, Dataset, FightSimulationRequest, FightSimulationResult,
  FootballSimulationRequest, FootballSimulationResult, QuantService, Sport,
} from './engine/types'
import { createService } from './services'
import { Backtest } from './components/Backtest'
import { FightSimulator } from './components/FightSimulator'
import { FootballSimulator } from './components/FootballSimulator'
import { MethodBreakdown } from './components/MethodBreakdown'
import { ScoreHeatmap } from './components/ScoreHeatmap'
import { TopBar, type TabId } from './components/TopBar'
import { Button } from './components/ui/Button'
import { EmptyState, ErrorState, Region, Skeleton, type RegionState } from './components/ui/States'
import { useHashParams } from './components/useHashParams'
import { useRunner } from './components/useRunner'
import { ValueScanner, type ScanInput } from './components/ValueScanner'

type Boot =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; service: QuantService; dataset: Dataset }

const TAB_IDS: TabId[] = ['sim', 'map', 'ev', 'bt']

function Shell({ mode, sport, tab, onSport, onTab, children }: {
  mode: 'api' | 'local' | null; sport: Sport; tab: TabId; onSport: (s: Sport) => void; onTab: (t: TabId) => void; children: ReactNode
}) {
  return (
    <>
      <a className="skip" href="#main">Saltar al contenido</a>
      <TopBar mode={mode} sport={sport} tab={tab} onSport={onSport} onTab={onTab} />
      <main id="main" className="shell">
        <h1 className="visually-hidden">QuantEdge Sports AI, simulador cuantitativo de fútbol y MMA</h1>
        {children}
      </main>
      <footer className="shell footer">
        <p>
          Los nombres de equipos y luchadores son reales, pero las estadísticas son aproximadas y el historial de resultados y cuotas es simulado.
          Esta herramienta es educativa y no constituye asesoramiento de apuestas.
        </p>
      </footer>
    </>
  )
}

function usePageState() {
  const [params, setParams] = useHashParams()
  const sport: Sport = params.get('s') === 'mma' ? 'mma' : 'football'
  const tab: TabId = TAB_IDS.find((t) => t === params.get('t')) ?? 'sim'
  const onSport = useCallback((s: Sport) => setParams({ s }, true), [setParams])
  const onTab = useCallback((t: TabId) => setParams({ t }, true), [setParams])
  return { params, setParams, sport, tab, onSport, onTab }
}

export default function App() {
  const page = usePageState()
  const [boot, setBoot] = useState<Boot>({ status: 'loading' })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let live = true
    setBoot({ status: 'loading' })
    createService()
      .then(async (service) => ({ service, dataset: await service.loadDataset() }))
      .then(({ service, dataset }) => { if (live) setBoot({ status: 'ready', service, dataset }) })
      .catch((e: unknown) => { if (live) setBoot({ status: 'error', message: e instanceof Error ? e.message : String(e) }) })
    return () => { live = false }
  }, [attempt])

  if (boot.status === 'ready') return <Workspace service={boot.service} dataset={boot.dataset} page={page} />

  return (
    <Shell mode={null} sport={page.sport} tab={page.tab} onSport={page.onSport} onTab={page.onTab}>
      {boot.status === 'error'
        ? <ErrorState message={boot.message} onRetry={() => setAttempt((a) => a + 1)} />
        : (
          <div className="workspace" aria-busy="true">
            <div className="panel controls"><Skeleton shape="bars" label="Cargando parámetros" /></div>
            <div className="results stack"><Skeleton shape="figures" /><Skeleton shape="chart" /><Skeleton shape="table" /></div>
          </div>
        )}
    </Shell>
  )
}

function Workspace({ service, dataset, page }: { service: QuantService; dataset: Dataset; page: ReturnType<typeof usePageState> }) {
  const { params, setParams, sport, tab, onSport, onTab } = page
  const football = useRunner<FootballSimulationRequest, FootballSimulationResult>((r) => service.simulateFootball(r))
  const fight = useRunner<FightSimulationRequest, FightSimulationResult>((r) => service.simulateFight(r))
  const backtest = useRunner<BacktestRequest, BacktestResult>((r) => service.backtestFootball(r))

  const active = sport === 'football' ? football : fight
  const scan: ScanInput = {
    status: active.status,
    error: active.error,
    matchup: active.data ? (sport === 'football' ? `${football.data!.prediction.home} contra ${football.data!.prediction.away}` : `${fight.data!.prediction.fighter_a} contra ${fight.data!.prediction.fighter_b}`) : null,
    valueBets: active.data?.value_bets ?? null,
    margin: active.data?.market_margin ?? null,
    bankroll: active.req?.bankroll ?? 1000,
    kelly: active.req?.kelly_multiplier ?? 0.25,
  }

  const panel = (id: TabId, children: ReactNode) => (
    <section role="tabpanel" id={`main-panel-${id}`} aria-labelledby={`main-tab-${id}`} hidden={tab !== id}>{children}</section>
  )
  const regionState = (r: { data: unknown; status: string }): RegionState => (r.data ? 'ready' : r.status === 'error' ? 'error' : 'loading')

  return (
    <Shell mode={service.mode} sport={sport} tab={tab} onSport={onSport} onTab={onTab}>
      {sport === 'football' ? (
        <>
          {panel('sim', <FootballSimulator dataset={dataset} runner={football} params={params} setParams={setParams} />)}
          {panel('map',
            <Region state={regionState(football)} error={football.error} skeleton={<div className="workspace workspace--wide"><Skeleton shape="grid" /><Skeleton shape="bars" /></div>}>
              {football.data ? <ScoreHeatmap result={football.data} /> : null}
            </Region>,
          )}
          {panel('ev', <ValueScanner scan={scan} onGoSimulator={() => onTab('sim')} />)}
          {panel('bt', <Backtest runner={backtest} active={tab === 'bt'} />)}
        </>
      ) : (
        <>
          {panel('sim', <FightSimulator dataset={dataset} runner={fight} params={params} setParams={setParams} />)}
          {panel('map',
            <Region state={regionState(fight)} error={fight.error} skeleton={<div className="stack"><Skeleton shape="table" /><Skeleton shape="bars" /></div>}>
              {fight.data ? <MethodBreakdown result={fight.data} /> : null}
            </Region>,
          )}
          {panel('ev', <ValueScanner scan={scan} onGoSimulator={() => onTab('sim')} />)}
          {panel('bt',
            <EmptyState title="El backtesting usa el historial de fútbol" action={<Button variant="secondary" onClick={() => onSport('football')}>Ir a fútbol</Button>}>
              El backtesting walk-forward evalúa el modelo partido a partido y corre sobre el historial de fútbol. Cambia de deporte para ejecutarlo.
            </EmptyState>,
          )}
        </>
      )}
    </Shell>
  )
}
