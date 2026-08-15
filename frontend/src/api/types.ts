// Types matching the FastAPI response schemas exactly

export interface Track {
  track_id: number
  title: string
  artist: string
  genre: string
  valence: number
  energy: number
  youtube_url: string
}

export interface SearchMeta {
  system: string
  target_valence: number
  target_energy: number
  query: string
}

export interface SearchResponse {
  tracks: Track[]
  meta: SearchMeta
}

export interface Waypoint {
  step: number
  valence: number
  energy: number
  tracks: Track[]
}

export interface RegulationMeta {
  current_mood: string
  target_mood: string
  current_coords: [number, number]
  target_coords: [number, number]
  n_waypoints: number
}

export interface RegulateResponse {
  waypoints: Waypoint[]
  meta: RegulationMeta
}