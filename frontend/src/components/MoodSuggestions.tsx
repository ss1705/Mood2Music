interface Props {
  onSelect: (mood: string) => void
}

const SUGGESTIONS = [
  "anxious and restless",
  "calm and reflective",
  "low energy",
  "happy and energized",
  "melancholy and tired",
  "nostalgic and bittersweet",
  "stressed and overwhelmed",
  "focused and driven",
]

export function MoodSuggestions({ onSelect }: Props) {
  return (
    <div style={{
      marginTop: '20px',
      display: 'flex',
      flexWrap: 'wrap',
      gap: '8px',
      justifyContent: 'center',
    }}>
      {SUGGESTIONS.map((mood) => (
        <button
          key={mood}
          onClick={() => onSelect(`I feel ${mood}`)}
          style={{
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: '20px',
            padding: '6px 14px',
            fontSize: '11px',
            color: 'rgba(255,255,255,0.4)',
            cursor: 'pointer',
            fontFamily: 'Inter, sans-serif',
            letterSpacing: '0.02em',
            transition: 'all 0.2s ease',
            whiteSpace: 'nowrap',
          }}
          onMouseEnter={e => {
            const el = e.currentTarget
            el.style.background = 'rgba(99,102,241,0.15)'
            el.style.borderColor = 'rgba(99,102,241,0.3)'
            el.style.color = 'rgba(199,199,255,0.8)'
          }}
          onMouseLeave={e => {
            const el = e.currentTarget
            el.style.background = 'rgba(255,255,255,0.04)'
            el.style.borderColor = 'rgba(255,255,255,0.08)'
            el.style.color = 'rgba(255,255,255,0.4)'
          }}
        >
          {mood}
        </button>
      ))}
    </div>
  )
}