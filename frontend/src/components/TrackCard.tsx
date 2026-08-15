import type { Track } from '../api/types'

interface Props {
  track: Track
  index?: number
}

export function TrackCard({ track, index }: Props) {
  const dotClass =
    track.valence > 0.6 ? 'dot-amber'
    : track.valence > 0.35 ? 'dot-teal'
    : 'dot-indigo'

  const energyBars = [0.25, 0.5, 0.75, 1.0]

  return (
    <div className="glass fade-in" style={{
      display: 'flex',
      alignItems: 'center',
      gap: '10px',
      padding: '12px 14px',
      marginBottom: '6px',
    }}>
      {index !== undefined && (
        <span style={{
          fontSize: '10px',
          color: 'rgba(255,255,255,0.15)',
          width: '14px',
          textAlign: 'right',
          flexShrink: 0,
        }}>
          {index + 1}
        </span>
      )}

      {/* Valence dot */}
      <div className={dotClass} style={{
        width: '7px',
        height: '7px',
        borderRadius: '50%',
        flexShrink: 0,
      }} />

      {/* Track info */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <p style={{
          fontSize: '12px',
          fontWeight: 500,
          color: 'rgba(255,255,255,0.82)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}>
          {track.title}
        </p>
        <p style={{
          fontSize: '10px',
          color: 'rgba(255,255,255,0.3)',
          marginTop: '2px',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}>
          {track.artist}
          {track.genre !== 'Unknown' && (
            <span style={{ color: 'rgba(255,255,255,0.15)', marginLeft: '6px' }}>
              · {track.genre}
            </span>
          )}
        </p>
      </div>

      {/* YouTube link */}
      <a    
      href={track.youtube_url}
      target="_blank"
      rel="noopener noreferrer"
      style={{
        flexShrink: 0,
        fontSize: '10px',
        color: 'rgba(255,255,255,0.2)',
        textDecoration: 'none',
        padding: '4px 8px',
        borderRadius: '6px',
        border: '1px solid rgba(255,255,255,0.08)',
        transition: 'all 0.2s ease',
        fontFamily: 'Inter, sans-serif',
        letterSpacing: '0.04em',
      }}
      onMouseEnter={e => {
        (e.target as HTMLElement).style.color = 'rgba(255,255,255,0.6)'
        ;(e.target as HTMLElement).style.borderColor = 'rgba(255,255,255,0.2)'
      }}
      onMouseLeave={e => {
        (e.target as HTMLElement).style.color = 'rgba(255,255,255,0.2)'
        ;(e.target as HTMLElement).style.borderColor = 'rgba(255,255,255,0.08)'
      }}
    >
      ▶ listen
    </a>

      {/* Energy bars */}
      <div style={{
        display: 'flex',
        gap: '2px',
        alignItems: 'flex-end',
        height: '14px',
        flexShrink: 0,
      }}>
        {energyBars.map((threshold, i) => (
          <div key={i} style={{
            width: '3px',
            height: `${(i + 1) * 3 + 2}px`,
            borderRadius: '2px',
            background: track.energy >= threshold
              ? 'rgba(255,255,255,0.4)'
              : 'rgba(255,255,255,0.08)',
          }} />
        ))}
      </div>
    </div>
  )
}