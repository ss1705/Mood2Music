import axios from 'axios'
import type { SearchResponse, RegulateResponse } from './types'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
})

export async function searchMood(query: string, k = 5): Promise<SearchResponse> {
  const { data } = await api.post<SearchResponse>('/search', { query, k })
  return data
}

export async function regulateMood(
  current_mood: string,
  target_mood: string,
  n_waypoints = 5
): Promise<RegulateResponse> {
  const { data } = await api.post<RegulateResponse>('/regulate', {
    current_mood,
    target_mood,
    n_waypoints,
  })
  return data
}