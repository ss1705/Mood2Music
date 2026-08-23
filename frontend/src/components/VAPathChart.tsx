import { useState } from 'react'
import type { RegulateResponse } from '../api/types'

interface Props {
  result: RegulateResponse
}

export function VAPathChart({ result }: Props) {
  const [hoveredWaypoint, setHoveredWaypoint] = useState<number | null>(null)
  const [hoveredStart, setHoveredStart] = useState(false)
  const [hoveredEnd, setHoveredEnd] = useState(false)

  const W = 340
  const H = 280
  const PAD = 36

  const plotW = W - PAD * 2
  const plotH = H - PAD * 2

  const toSVG = (valence: number, energy: number) => ({
    x: PAD + valence * plotW,
    y: PAD + (1 - energy) * plotH,
  })

  const current = toSVG(result.meta.current_coords[0], result.meta.current_coords[1])
  const target = toSVG(result.meta.target_coords[0], result.meta.target_coords[1])
  const waypoints = result.waypoints.map(wp => toSVG(wp.valence, wp.energy))

  const pathD = waypoints
    .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${pt.x.toFixed(1)} ${pt.y.toFixed(1)}`)
    .join(' ')

  return (
    <div style={{ marginTop: '24px', marginBottom: '8px' }}>
      <p style={{
        fontSize: '10px',
        color: 'rgba(255,255,255,0.2)',
        letterSpacing: '0.08em',
        marginBottom: '12px',
        textAlign: 'center',
      }}>
        your emotional path
      </p>

      <div style={{ display: 'flex', justifyContent: 'center', position: 'relative' }}>
        <svg
          width={W}
          height={H}
          style={{
            background: 'rgba(255,255,255,0.02)',
            borderRadius: '14px',
            border: '1px solid rgba(255,255,255,0.06)',
            overflow: 'visible',
          }}
        >
          {/* Quadrant labels */}
          <text x={PAD + 6} y={PAD + 14} fontSize="9" fill="rgba(255,255,255,0.12)" fontFamily="Inter,sans-serif">tense</text>
          <text x={W - PAD - 6} y={PAD + 14} fontSize="9" fill="rgba(255,255,255,0.12)" fontFamily="Inter,sans-serif" textAnchor="end">excited</text>
          <text x={PAD + 6} y={H - PAD - 8} fontSize="9" fill="rgba(255,255,255,0.12)" fontFamily="Inter,sans-serif">sad</text>
          <text x={W - PAD - 6} y={H - PAD - 8} fontSize="9" fill="rgba(255,255,255,0.12)" fontFamily="Inter,sans-serif" textAnchor="end">calm</text>

          {/* Axis labels */}
          <text x={W / 2} y={H - 6} fontSize="8" fill="rgba(255,255,255,0.15)" fontFamily="Inter,sans-serif" textAnchor="middle">valence →</text>
          <text x={10} y={H / 2} fontSize="8" fill="rgba(255,255,255,0.15)" fontFamily="Inter,sans-serif" textAnchor="middle" transform={`rotate(-90, 10, ${H / 2})`}>energy →</text>

          {/* Quadrant dividers */}
          <line x1={PAD + plotW / 2} y1={PAD} x2={PAD + plotW / 2} y2={PAD + plotH} stroke="rgba(255,255,255,0.05)" strokeWidth="1" strokeDasharray="3 3" />
          <line x1={PAD} y1={PAD + plotH / 2} x2={PAD + plotW} y2={PAD + plotH / 2} stroke="rgba(255,255,255,0.05)" strokeWidth="1" strokeDasharray="3 3" />

          {/* Path */}
          <path d={pathD} fill="none" stroke="rgba(99,102,241,0.35)" strokeWidth="1.5" strokeDasharray="5 3" />

          {/* Intermediate waypoint dots — hoverable */}
          {result.waypoints.slice(1, -1).map((_wp, i) => {
            const pt = waypoints[i + 1]
            const isHovered = hoveredWaypoint === i + 1
            return (
              <g key={i}>
                <circle
                  cx={pt.x}
                  cy={pt.y}
                  r={isHovered ? 10 : 6}
                  fill="rgba(99,102,241,0.15)"
                  style={{ cursor: 'pointer', transition: 'r 0.2s' }}
                  onMouseEnter={() => setHoveredWaypoint(i + 1)}
                  onMouseLeave={() => setHoveredWaypoint(null)}
                />
                <circle
                  cx={pt.x}
                  cy={pt.y}
                  r={isHovered ? 5 : 3}
                  fill={isHovered ? '#818cf8' : 'rgba(99,102,241,0.6)'}
                  style={{ cursor: 'pointer', pointerEvents: 'none' }}
                />
                {/* Step number */}
                <text
                  x={pt.x}
                  y={pt.y - 10}
                  fontSize="8"
                  fill="rgba(255,255,255,0.25)"
                  textAnchor="middle"
                  fontFamily="Inter,sans-serif"
                >
                  {i + 2}
                </text>
              </g>
            )
          })}

          {/* Current mood dot — indigo */}
           <g
            onMouseEnter={() => setHoveredStart(true)}
            onMouseLeave={() => setHoveredStart(false)}
            style={{ cursor: 'pointer' }}
            >
            <circle cx={current.x} cy={current.y} r={hoveredStart ? 14 : 10} fill="rgba(99,102,241,0.15)" style={{ transition: 'r 0.2s' }} />
            <circle cx={current.x} cy={current.y} r={hoveredStart ? 7 : 5} fill="#818cf8" style={{ pointerEvents: 'none' }} />
            <text x={current.x} y={current.y - 14} fontSize="9" fill="rgba(199,199,255,0.8)" textAnchor="middle" fontFamily="Inter,sans-serif" fontWeight="500">now</text>
            </g>
            
           {/* Target mood dot — teal */}
            <g
            onMouseEnter={() => setHoveredEnd(true)}
            onMouseLeave={() => setHoveredEnd(false)}
            style={{ cursor: 'pointer' }}
            >
            <circle cx={target.x} cy={target.y} r={hoveredEnd ? 14 : 10} fill="rgba(45,212,191,0.15)" style={{ transition: 'r 0.2s' }} />
            <circle cx={target.x} cy={target.y} r={hoveredEnd ? 7 : 5} fill="#2dd4bf" style={{ pointerEvents: 'none' }} />
            <text x={target.x} y={target.y - 14} fontSize="9" fill="rgba(153,246,228,0.8)" textAnchor="middle" fontFamily="Inter,sans-serif" fontWeight="500">goal</text>
            </g>
        </svg>

        {/* Hover tooltip — shows tracks at hovered waypoint */}
        {hoveredWaypoint !== null && (
          <div style={{
            position: 'absolute',
            top: '8px',
            right: '-10px',
            background: 'rgba(13,1,24,0.95)',
            border: '1px solid rgba(99,102,241,0.3)',
            borderRadius: '10px',
            padding: '10px 14px',
            width: '180px',
            pointerEvents: 'none',
          }}>
            <p style={{ fontSize: '9px', color: 'rgba(199,199,255,0.6)', marginBottom: '6px', letterSpacing: '0.06em' }}>
              step {hoveredWaypoint}
            </p>
            {result.waypoints[hoveredWaypoint]?.tracks.map((track, i) => (
              <div key={i} style={{ marginBottom: '4px' }}>
                <p style={{ fontSize: '10px', color: 'rgba(255,255,255,0.75)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {track.title}
                </p>
                <p style={{ fontSize: '9px', color: 'rgba(255,255,255,0.3)' }}>
                  {track.artist}
                </p>
              </div>
            ))}
          </div>
        )}

        {/* Start tooltip */}
        {hoveredStart && (
        <div style={{
            position: 'absolute',
            top: '8px',
            left: '-10px',
            background: 'rgba(13,1,24,0.95)',
            border: '1px solid rgba(99,102,241,0.3)',
            borderRadius: '10px',
            padding: '10px 14px',
            width: '180px',
            pointerEvents: 'none',
        }}>
            <p style={{ fontSize: '9px', color: 'rgba(199,199,255,0.6)', marginBottom: '6px', letterSpacing: '0.06em' }}>
            where you are
            </p>
            {result.waypoints[0]?.tracks.map((track, i) => (
            <div key={i} style={{ marginBottom: '4px' }}>
                <p style={{ fontSize: '10px', color: 'rgba(255,255,255,0.75)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {track.title}
                </p>
                <p style={{ fontSize: '9px', color: 'rgba(255,255,255,0.3)' }}>
                {track.artist}
                </p>
            </div>
            ))}
            <p style={{ fontSize: '8px', color: 'rgba(199,199,255,0.3)', marginTop: '6px' }}>
            val {result.meta.current_coords[0].toFixed(2)} · nrg {result.meta.current_coords[1].toFixed(2)}
            </p>
        </div>
        )}

        {/* End tooltip */}
        {hoveredEnd && (
        <div style={{
            position: 'absolute',
            top: '8px',
            right: '-10px',
            background: 'rgba(13,1,24,0.95)',
            border: '1px solid rgba(45,212,191,0.3)',
            borderRadius: '10px',
            padding: '10px 14px',
            width: '180px',
            pointerEvents: 'none',
        }}>
            <p style={{ fontSize: '9px', color: 'rgba(153,246,228,0.6)', marginBottom: '6px', letterSpacing: '0.06em' }}>
            where you're going
            </p>
            {result.waypoints[result.waypoints.length - 1]?.tracks.map((track, i) => (
            <div key={i} style={{ marginBottom: '4px' }}>
                <p style={{ fontSize: '10px', color: 'rgba(255,255,255,0.75)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {track.title}
                </p>
                <p style={{ fontSize: '9px', color: 'rgba(255,255,255,0.3)' }}>
                {track.artist}
                </p>
            </div>
            ))}
            <p style={{ fontSize: '8px', color: 'rgba(153,246,228,0.3)', marginTop: '6px' }}>
            val {result.meta.target_coords[0].toFixed(2)} · nrg {result.meta.target_coords[1].toFixed(2)}
            </p>
        </div>
        )}
      </div>
    </div>
  )
}