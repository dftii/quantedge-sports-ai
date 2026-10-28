import { ChartLine, Cpu, Dices, Grid3x3, Loader, ScanSearch, Server } from 'lucide-react'
import type { Sport } from '../engine/types'
import { Tabs, type TabItem } from './ui/Tabs'

export type TabId = 'sim' | 'map' | 'ev' | 'bt'

const SPORTS: TabItem<Sport>[] = [
  { id: 'football', label: 'Fútbol' },
  { id: 'mma', label: 'MMA' },
]

const TABS: TabItem<TabId>[] = [
  { id: 'sim', label: 'Simulador H2H', icon: <Dices size={16} aria-hidden="true" /> },
  { id: 'map', label: 'Mapa de calor', icon: <Grid3x3 size={16} aria-hidden="true" /> },
  { id: 'ev', label: 'Escáner +EV', icon: <ScanSearch size={16} aria-hidden="true" /> },
  { id: 'bt', label: 'Backtesting', icon: <ChartLine size={16} aria-hidden="true" /> },
]

interface Props {
  /** null while the service is still resolving */
  mode: 'api' | 'local' | null
  sport: Sport
  tab: TabId
  onSport: (s: Sport) => void
  onTab: (t: TabId) => void
}

function Source({ mode }: { mode: Props['mode'] }) {
  if (mode === null) return <p className="source"><Loader className="spin" size={16} aria-hidden="true" /><span>Conectando con el motor</span></p>
  return (
    <p className="source">
      {mode === 'api' ? <Server size={16} aria-hidden="true" /> : <Cpu size={16} aria-hidden="true" />}
      <span>{mode === 'api' ? 'API local' : 'Motor en navegador'}</span>
      <span className="muted">{mode === 'api' ? 'FastAPI' : 'TypeScript'}</span>
    </p>
  )
}

export function TopBar({ mode, sport, tab, onSport, onTab }: Props) {
  return (
    <header className="topbar">
      <div className="shell topbar__row">
        <a className="wordmark" href="#" aria-label="QuantEdge Sports AI, inicio" onClick={(e) => { e.preventDefault(); onTab('sim') }}>
          quantedge<span className="wordmark__caret" aria-hidden="true" />
        </a>
        <Tabs items={SPORTS} value={sport} onChange={onSport} label="Deporte" idPrefix="sport" panels={false} className="tabs--sport" />
        <Source mode={mode} />
      </div>
      <div className="shell">
        <Tabs items={TABS} value={tab} onChange={onTab} label="Secciones" idPrefix="main" className="tabs--main" />
      </div>
    </header>
  )
}
