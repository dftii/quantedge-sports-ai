import dataset from '../data/dataset.json'
import type { Dataset, QuantService } from '../engine/types'
import { createApiService } from './apiService'
import { createLocalService } from './localService'

const HEALTH_TIMEOUT_MS = 1500

/** GitHub Pages (or VITE_FORCE_LOCAL=1) runs the in-browser engines; otherwise the FastAPI backend when it answers /api/health. */
export async function createService(): Promise<QuantService> {
  const local = () => createLocalService(dataset as unknown as Dataset)
  if (import.meta.env.VITE_FORCE_LOCAL === '1' || location.hostname.endsWith('.github.io')) return local()
  const baseUrl: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
  try {
    const res = await fetch(`${baseUrl.replace(/\/+$/, '')}/api/health`, { signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS) })
    if (!res.ok) throw new Error(`health ${res.status}`)
    return createApiService(baseUrl)
  } catch {
    return local()
  }
}
