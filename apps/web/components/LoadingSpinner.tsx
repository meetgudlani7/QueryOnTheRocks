'use client'

import { useEffect, useState } from 'react'

interface LoadingSpinnerProps {
  mode?: 'stt' | 'retrieve' | 'generate' | 'verify' | 'generic'
}

const MESSAGES: Record<NonNullable<LoadingSpinnerProps['mode']>, string[]> = {
  stt:      ['Transcribing your voice…', 'Whisper is listening…'],
  retrieve: ['Searching the knowledge base…', 'Running hybrid retrieval…', 'Fusing BM25 + dense results…'],
  generate: ['Generating answer…', 'Reading the evidence…', 'Writing a grounded response…'],
  verify:   ['Checking groundedness…', 'Validating citations…', 'Fact-checking the answer…'],
  generic:  ['Processing…', 'Thinking…'],
}

export default function LoadingSpinner({ mode = 'generic' }: LoadingSpinnerProps) {
  const [msgIndex, setMsgIndex] = useState(0)
  const messages = MESSAGES[mode]

  useEffect(() => {
    setMsgIndex(0)
    if (messages.length <= 1) return
    const id = setInterval(() => {
      setMsgIndex((i) => (i + 1) % messages.length)
    }, 1800)
    return () => clearInterval(id)
  }, [mode])

  return (
    <div
      role="status"
      aria-live="polite"
      aria-label={messages[msgIndex]}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '0.75rem',
      }}
    >
      {/* Spinner ring */}
      <div style={{
        width: '32px',
        height: '32px',
        border: '3px solid rgba(75,158,191,0.2)',
        borderTop: '3px solid var(--ocean)',
        borderRadius: '50%',
        animation: 'spin 0.9s linear infinite',
      }} aria-hidden="true" />

      {/* Message */}
      <p style={{
        fontSize: '0.78rem',
        fontWeight: 600,
        letterSpacing: '0.06em',
        color: 'var(--charcoal)',
        opacity: 0.6,
        margin: 0,
        transition: 'opacity 0.3s',
      }}>
        {messages[msgIndex]}
      </p>
    </div>
  )
}