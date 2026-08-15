import { useState } from 'react'
import type { SearchMeta, RegulationMeta } from '../api/types'

type Props =
  | { type: 'search'; meta: SearchMeta }
  | { type: 'regulate'; meta: RegulationMeta }

export function UnderTheHood(props: Props) {
  const [open, setOpen] = useState(false)

  return (
    <div style={{ marginTop: '16px' }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontSize: '10px',
          color: 'rgba(255,255,255,0.22)',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          fontFamily: 'Inter, sans-serif',
          padding: '0',
          letterSpacing: '0.02em',
        }}
      >
        <span>{open ? '▾' : '▸'}</span>
        <span>how was this generated?</span>
      </button>

      {open && (
        <div className="fade-in" style={{
          marginTop: '8px',
          background: 'rgba(255,255,255,0.02)',
          border: '1px solid rgba(255,255,255,0.05)',
          borderRadius: '10px',
          padding: '12px 14px',
        }}>
          {props.type === 'search' && (
            <>
              <Row label="system" value={
                props.meta.system === 'hybrid_clap'
                  ? 'CLAP audio embeddings + valence/energy reranker'
                  : 'sentence-transformer text retrieval'
              } />
              <Row label="mapped to" value={
                `valence ${props.meta.target_valence.toFixed(2)},
                 energy ${props.meta.target_energy.toFixed(2)}`
              } />
              <p style={{
                fontSize: '10px',
                color: 'rgba(255,255,255,0.22)',
                lineHeight: 1.6,
                marginTop: '8px',
              }}>
                Your mood was placed in valence-arousal space — a 2D model of
                emotion from music psychology (Russell, 1980). Tracks were
                retrieved whose profile sits closest to that position.
              </p>
            </>
          )}

          {props.type === 'regulate' && (
            <>
              <Row label="from" value={
                `valence ${props.meta.current_coords[0].toFixed(2)},
                 energy ${props.meta.current_coords[1].toFixed(2)}`
              } />
              <Row label="to" value={
                `valence ${props.meta.target_coords[0].toFixed(2)},
                 energy ${props.meta.target_coords[1].toFixed(2)}`
              } />
              <Row label="method" value={
                `NRC VAD lexicon → ${props.meta.n_waypoints} waypoints interpolated linearly`
              } />
              <p style={{
                fontSize: '10px',
                color: 'rgba(255,255,255,0.22)',
                lineHeight: 1.6,
                marginTop: '8px',
              }}>
                Both moods were placed in valence-arousal space using the NRC
                VAD Lexicon — 20,000 words rated by humans on emotional
                dimensions. Waypoints were interpolated along the path between
                them. At each step, tracks were retrieved by Euclidean distance
                in that space.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div style={{
      display: 'flex',
      gap: '8px',
      fontSize: '10px',
      marginBottom: '4px',
    }}>
      <span style={{ color: 'rgba(255,255,255,0.45)', flexShrink: 0 }}>
        {label}
      </span>
      <span style={{ color: 'rgba(255,255,255,0.3)' }}>{value}</span>
    </div>
  )
}