import { useState, useEffect } from 'react'
import { MoodInput } from './components/MoodInput'
import { TrackCard } from './components/TrackCard'
import { UnderTheHood } from './components/UnderTheHood'
import { searchMood, regulateMood } from './api/client'
import type { SearchResponse, RegulateResponse } from './api/types'

// Derive glow configuration from average valence/energy of results
function getGlowConfig(valence: number, energy: number) {
  if (valence < 0.35 && energy > 0.55) {
    // stressed / tense — indigo top right, faint teal bottom left
    return {
      g1: { color: 'indigo', size: 400, top: -100, right: -80, opacity: 1 },
      g2: { color: 'teal',   size: 250, bottom: 100, left: -60, opacity: 0.6 },
    }
  } else if (valence < 0.35 && energy <= 0.55) {
    // melancholy / tired — deep indigo
    return {
      g1: { color: 'indigo', size: 500, top: -120, right: -100, opacity: 1 },
      g2: { color: 'indigo', size: 200, bottom: 150, left: -40, opacity: 0.4 },
    }
  } else if (valence >= 0.6 && energy > 0.6) {
    // happy / energized — amber
    return {
      g1: { color: 'amber', size: 400, top: -80, right: -60, opacity: 1 },
      g2: { color: 'amber', size: 250, bottom: 80, left: -40, opacity: 0.5 },
    }
  } else if (valence >= 0.6 && energy <= 0.4) {
    // calm / peaceful — teal
    return {
      g1: { color: 'teal', size: 450, top: -100, right: -80, opacity: 1 },
      g2: { color: 'teal', size: 200, bottom: 100, left: -30, opacity: 0.5 },
    }
  } else {
    // neutral / mixed — soft indigo + teal
    return {
      g1: { color: 'indigo', size: 350, top: -80, right: -60, opacity: 0.7 },
      g2: { color: 'teal',   size: 250, bottom: 80, left: -40, opacity: 0.5 },
    }
  }
}

export default function App() {
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null)
  const [regulationResults, setRegulationResults] = useState<RegulateResponse | null>(null)
  const [regulationMode, setRegulationMode] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [currentMood, setCurrentMood] = useState('')
  const [submittedMood, setSubmittedMood] = useState('')
  const [glowConfig, setGlowConfig] = useState<ReturnType<typeof getGlowConfig> | null>(null)

  // Update glow when results change
  useEffect(() => {
    if (searchResults) {
      const avgValence = searchResults.tracks.reduce((s, t) => s + t.valence, 0) / searchResults.tracks.length
      const avgEnergy = searchResults.tracks.reduce((s, t) => s + t.energy, 0) / searchResults.tracks.length
      setGlowConfig(getGlowConfig(avgValence, avgEnergy))
    }
  }, [searchResults])

  useEffect(() => {
    if (regulationResults) {
      const firstWp = regulationResults.waypoints[0]
      const lastWp = regulationResults.waypoints[regulationResults.waypoints.length - 1]
      const firstTrack = firstWp.tracks[0]
      const lastTrack = lastWp.tracks[0]
      if (firstTrack && lastTrack) {
        setGlowConfig({
          g1: getGlowConfig(firstTrack.valence, firstTrack.energy).g1,
          g2: getGlowConfig(lastTrack.valence, lastTrack.energy).g2,
        })
      }
    }
  }, [regulationResults])

  async function handleSearch(query: string) {
    setLoading(true)
    setError(null)
    setCurrentMood(query)
    setSubmittedMood(query)
    setRegulationResults(null)
    setRegulationMode(false)
    try {
      const results = await searchMood(query)
      setSearchResults(results)
    } catch {
      setError('something went wrong. please try again.')
    } finally {
      setLoading(false)
    }
  }

  async function handleRegulate(targetMood: string) {
    if (!currentMood) return
    setLoading(true)
    setError(null)
    try {
      const results = await regulateMood(currentMood, targetMood)
      setRegulationResults(results)
    } catch {
      setError('something went wrong. please try again.')
    } finally {
      setLoading(false)
    }
  }

  function reset() {
    setSearchResults(null)
    setRegulationResults(null)
    setRegulationMode(false)
    setCurrentMood('')
    setSubmittedMood('')
    setError(null)
    setGlowConfig(null)
  }

  const g = glowConfig

  return (
    <>
      {/* Reactive glow layer */}
      <div className="glow-layer">
        {g && (
          <>
            <div
              className={`glow glow-${g.g1.color}`}
              style={{
                width: g.g1.size,
                height: g.g1.size,
                top: g.g1.top,
                right: 'right' in g.g1 ? g.g1.right : undefined,
                left: 'left' in g.g1 ? (g.g1 as any).left : undefined,
                opacity: g.g1.opacity,
              }}
            />
            <div
              className={`glow glow-${g.g2.color}`}
              style={{
                width: g.g2.size,
                height: g.g2.size,
                bottom: 'bottom' in g.g2 ? g.g2.bottom : undefined,
                top: 'top' in g.g2 ? (g.g2 as any).top : undefined,
                left: 'left' in g.g2 ? g.g2.left : undefined,
                right: 'right' in g.g2 ? (g.g2 as any).right : undefined,
                opacity: g.g2.opacity,
              }}
            />
          </>
        )}
      </div>

      {/* Main content */}
      <div className="content" style={{
        minHeight: '100vh',
        padding: '72px 16px 80px',
        maxWidth: '580px',
        margin: '0 auto',
      }}>

        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '52px' }}>
          <h1 style={{
            fontSize: '62px',
            fontWeight: 500,
            fontStyle: 'italic',
            fontFamily: "'Cormorant Garamond', serif",
            color: 'rgba(255,255,255,0.85)',
            marginBottom: '6px',
            lineHeight: 1,
            letterSpacing: '0.01em',
          }}>
            Mood2Music
          </h1>
          <p style={{
            fontSize: '12px',
            color: 'rgba(210, 203, 203, 0.65)',
            letterSpacing: '0.1em',
            fontFamily: "'Inter', serif",
            fontWeight: 400,
          }}>
            music for how you feel
          </p>
        </div>

        {/* Initial input */}
        {!submittedMood && (
          <MoodInput
            placeholder="I feel..."
            onSubmit={handleSearch}
            loading={loading}
            autoFocus
          />
        )}

        {/* Submitted mood tag */}
        {submittedMood && !regulationResults && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            marginBottom: '20px',
          }}>
            <span style={{
              fontSize: '12px',
              color: 'rgba(255,255,255,0.38)',
              fontStyle: 'italic',
            }}>
              "{submittedMood}"
            </span>
            <button onClick={reset} style={{
              fontSize: '10px',
              color: 'rgba(255,255,255,0.18)',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              textDecoration: 'underline',
              textDecorationColor: 'rgba(255,255,255,0.1)',
              textUnderlineOffset: '3px',
              fontFamily: 'Inter, sans-serif',
            }}>
              change
            </button>
          </div>
        )}

        {/* Regulation mood summary */}
        {submittedMood && regulationResults && (
          <div style={{ marginBottom: '20px' }}>
            <span style={{
              fontSize: '12px',
              color: 'rgba(255,255,255,0.35)',
              fontStyle: 'italic',
            }}>
              "{submittedMood}"
              <span style={{ margin: '0 8px', color: 'rgba(255,255,255,0.15)' }}>→</span>
              "{regulationResults.meta.target_mood}"
            </span>
          </div>
        )}

        {error && (
          <p style={{
            fontSize: '11px',
            color: 'rgba(220,80,80,0.65)',
            textAlign: 'center',
            marginBottom: '16px',
          }}>
            {error}
          </p>
        )}

        {/* Loading */}
        {loading && (
          <div className="fade-in" style={{
            textAlign: 'center',
            padding: '56px 0',
          }}>
            <p style={{
              fontSize: '11px',
              color: 'rgba(255,255,255,0.18)',
              letterSpacing: '0.1em',
            }}>
              finding your music
            </p>
          </div>
        )}

        {/* Search results */}
        {searchResults && !regulationResults && !loading && (
          <div className="fade-in">
            {searchResults.tracks.map((track, i) => (
              <TrackCard key={track.track_id} track={track} index={i} />
            ))}

            <UnderTheHood type="search" meta={searchResults.meta} />

            {!regulationMode && (
              <div className="fade-in" style={{
                textAlign: 'center',
                marginTop: '36px',
              }}>
                <p style={{
                  fontSize: '11px',
                  color: 'rgba(255,255,255,0.18)',
                  marginBottom: '10px',
                }}>
                  want to shift how you feel?
                </p>
                <button
                  onClick={() => setRegulationMode(true)}
                  style={{
                    fontSize: '11px',
                    color: 'rgba(255,255,255,0.35)',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    textDecoration: 'underline',
                    textDecorationColor: 'rgba(255,255,255,0.12)',
                    textUnderlineOffset: '4px',
                    fontFamily: 'Inter, sans-serif',
                    letterSpacing: '0.02em',
                  }}
                >
                  take me somewhere else
                </button>
              </div>
            )}

            {regulationMode && (
              <div className="fade-in" style={{ marginTop: '28px' }}>
                <MoodInput
                  placeholder="I want to feel..."
                  onSubmit={handleRegulate}
                  loading={loading}
                  label="where do you want to go?"
                  autoFocus
                />
              </div>
            )}
          </div>
        )}

        {/* Regulation results */}
        {regulationResults && !loading && (
          <div className="fade-in">
            <p style={{
              fontSize: '10px',
              color: 'rgba(255,255,255,0.18)',
              textAlign: 'center',
              marginBottom: '4px',
              letterSpacing: '0.1em',
            }}>
              your journey
            </p>

            {regulationResults.waypoints.map((wp, wi) => {
              const isFirst = wi === 0
              const isLast = wi === regulationResults.waypoints.length - 1
              const label = isFirst ? 'where you are'
                : isLast ? "where you're going"
                : null

              return (
                <div key={wi}>
                  <div className="wp-divider">
                    <div className="wp-line" />
                    <span style={{
                      fontSize: '9px',
                      letterSpacing: '0.07em',
                      whiteSpace: 'nowrap',
                      color: isFirst || isLast
                        ? 'rgba(255,255,255,0.32)'
                        : 'rgba(255,255,255,0.16)',
                      fontWeight: isFirst || isLast ? 500 : 400,
                    }}>
                      {label ?? `step ${wi}`}
                    </span>
                    <div className="wp-line" />
                  </div>

                  {wp.tracks.map(track => (
                    <TrackCard key={track.track_id} track={track} />
                  ))}
                </div>
              )
            })}

            <UnderTheHood type="regulate" meta={regulationResults.meta} />

            <div style={{ marginTop: '48px', textAlign: 'center' }}>
              <button onClick={reset} style={{
                fontSize: '10px',
                color: 'rgba(255,255,255,0.18)',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                fontFamily: 'Inter, sans-serif',
                letterSpacing: '0.04em',
              }}>
                start over
              </button>
            </div>
          </div>
        )}
      </div>
    </>
  )
}