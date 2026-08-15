import { useState, useRef, useEffect } from 'react'

interface Props {
  placeholder: string
  onSubmit: (value: string) => void
  loading?: boolean
  autoFocus?: boolean
  label?: string
}

export function MoodInput({ placeholder, onSubmit, loading, autoFocus, label }: Props) {
  const [value, setValue] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (autoFocus) setTimeout(() => ref.current?.focus(), 120)
  }, [autoFocus])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (value.trim() && !loading) onSubmit(value.trim())
    }
  }

  return (
    <div>
      {label && (
        <p style={{
          fontSize: '10px',
          color: 'rgba(255,255,255,0.22)',
          textAlign: 'center',
          marginBottom: '10px',
          letterSpacing: '0.06em',
        }}>
          {label}
        </p>
      )}
      <div className="mood-input-wrap">
        <textarea
          ref={ref}
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={2}
          disabled={loading}
          style={{ opacity: loading ? 0.5 : 1 }}
        />
        <button
          onClick={() => value.trim() && !loading && onSubmit(value.trim())}
          disabled={!value.trim() || loading}
          style={{
            position: 'absolute',
            bottom: '12px',
            right: '12px',
            padding: '5px 10px',
            borderRadius: '20px',
            fontSize: '11px',
            background: value.trim() && !loading
              ? 'rgba(99,102,241,0.25)'
              : 'transparent',
            color: value.trim() && !loading
              ? 'rgba(199,199,255,0.7)'
              : 'rgba(255,255,255,0.15)',
            border: '1px solid',
            borderColor: value.trim() && !loading
              ? 'rgba(99,102,241,0.3)'
              : 'transparent',
            cursor: value.trim() && !loading ? 'pointer' : 'default',
            transition: 'all 0.2s ease',
            fontFamily: 'Inter, sans-serif',
          }}
        >
          {loading ? '· · ·' : '↵'}
        </button>
      </div>
    </div>
  )
}