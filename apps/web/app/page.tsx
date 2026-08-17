'use client'

import { useState } from 'react'
import AudioRecorder from '../components/AudioRecorder'
import AnswerDisplay from '../components/AnswerDisplay'
import LoadingSpinner from '../components/LoadingSpinner'
import PipelineIndicator, { buildStages, PipelineMode } from '../components/PipelineIndicator'
import { queryAPIStream, uploadAudio, StreamEvent } from '../lib/api'

const EXAMPLES = [
  'Who invented the telephone?',
  'What causes the northern lights?',
  'How does a vaccine work?',
  'What is the speed of light?',
]

export default function Home() {
  const [query, setQuery] = useState('')
  const [answer, setAnswer] = useState('')
  const [evidence, setEvidence] = useState<string[]>([])
  const [confidence, setConfidence] = useState(0)
  const [grounded, setGrounded] = useState(false)
  const [language, setLanguage] = useState('')
  const [latency, setLatency] = useState(0)
  const [sttLatency, setSttLatency] = useState<number | undefined>(undefined)
  const [isLoading, setIsLoading] = useState(false)
  const [verifying, setVerifying] = useState(false)
  const [unverifiedReason, setUnverifiedReason] = useState('')
  const [error, setError] = useState('')
  const [isVoice, setIsVoice] = useState(false)
  const [hasQueried, setHasQueried] = useState(false)

  const pipelineMode: PipelineMode = (() => {
    if (error) return 'error'
    if (verifying) return 'verifying'
    if (isLoading) return 'loading'
    if (answer || evidence.length > 0) return 'complete'
    return 'idle'
  })()

  const loadingMode = (() => {
    if (!isLoading) return 'generic'
    if (isVoice && !answer) return 'stt'
    return 'retrieve'
  })() as 'stt' | 'retrieve' | 'generate' | 'verify' | 'generic'

  const showPipeline = pipelineMode !== 'idle'
  const hasResult = answer || evidence.length > 0
  const showEmpty = !hasQueried && !isLoading && !error

  const handleTextQuery = async (overrideQuery?: string) => {
    const q = overrideQuery ?? query
    if (!q.trim()) return

    setIsVoice(false)
    setHasQueried(true)
    setIsLoading(true)
    setError('')
    setSttLatency(undefined)
    setAnswer('')
    setEvidence([])
    setConfidence(0)
    setGrounded(false)
    setUnverifiedReason('')
    setVerifying(false)

    let streamedAnswer = ''

    const onEvent = (event: StreamEvent) => {
      switch (event.type) {
        case 'token':
          streamedAnswer += event.text || ''
          setAnswer(streamedAnswer)
          setIsLoading(false)
          setVerifying(true)
          break
        case 'refused':
          setAnswer(
            event.reason ||
            "I couldn't find enough information in the knowledge base to answer that."
          )
          setEvidence(event.evidence || [])
          setConfidence(event.confidence || 0)
          setVerifying(false)
          setIsLoading(false)
          break
        case 'unverified':
          setEvidence(event.evidence || [])
          setUnverifiedReason(
            event.reason || 'This answer could not be confirmed as grounded.'
          )
          setVerifying(false)
          setIsLoading(false)
          break
        case 'verified':
          setEvidence(event.evidence || [])
          setConfidence(event.confidence || 0)
          setGrounded(event.grounded || false)
          setLatency(event.latency_ms || 0)
          setLanguage('en')
          setVerifying(false)
          setIsLoading(false)
          break
        case 'error':
          setError(event.message || 'Streaming query failed')
          setVerifying(false)
          setIsLoading(false)
          break
      }
    }

    try {
      await queryAPIStream({ query: q, language: 'en' }, onEvent)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setIsLoading(false)
      setVerifying(false)
    }
  }

  const handleAudioSubmitted = async (audioBlob: Blob) => {
    setIsVoice(true)
    setHasQueried(true)
    setIsLoading(true)
    setError('')
    setQuery('')
    setAnswer('')
    setEvidence([])
    setConfidence(0)
    setGrounded(false)
    setUnverifiedReason('')
    setVerifying(false)

    try {
      const data = await uploadAudio(audioBlob)
      setQuery(data.transcript || '')
      setAnswer(data.answer || '')
      setEvidence(data.evidence || [])
      setConfidence(data.confidence || 0)
      setGrounded(data.grounded || false)
      setLanguage(data.language || 'en')
      setLatency(data.rag_latency_ms || 0)
      setSttLatency(data.stt_latency_ms)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setIsLoading(false)
    }
  }

  const handleExample = (q: string) => {
    setQuery(q)
    handleTextQuery(q)
  }

  return (
    <div className="page-shell">
      <div className="content-col">

        {/* ── HEADER ─────────────────────────────────────────────── */}
        <header style={{ textAlign: 'center', padding: '1.25rem 0 0.25rem' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.5rem',
            marginBottom: '0.3rem',
          }}>
            <span style={{ fontSize: '1.5rem' }}>🪨</span>
            <h1 style={{
              fontSize: '1.75rem',
              fontWeight: 800,
              letterSpacing: '-0.03em',
              color: '#fff',
              textShadow: '0 2px 12px rgba(0,0,0,0.5), 0 1px 3px rgba(0,0,0,0.4)',
              margin: 0,
            }}>
              QueryOnTheRocks
            </h1>
          </div>
          <p style={{
            fontSize: '0.72rem',
            fontWeight: 600,
            letterSpacing: '0.18em',
            textTransform: 'uppercase',
            color: 'rgba(253,248,238,0.65)',
            textShadow: '0 1px 4px rgba(0,0,0,0.5)',
            margin: '0 0 0.5rem',
          }}>
            Voice-powered RAG · Goa 2026
          </p>
          <span style={{
            display: 'inline-block',
            fontSize: '0.58rem',
            fontWeight: 700,
            letterSpacing: '0.16em',
            textTransform: 'uppercase',
            color: 'var(--mustard)',
            background: 'rgba(0,0,0,0.45)',
            border: '1px solid rgba(243,186,32,0.3)',
            padding: '0.22rem 0.65rem',
            borderRadius: '20px',
            backdropFilter: 'blur(8px)',
          }}>
            HH GOA 2026
          </span>
        </header>

        {/* ── QUERY COMPOSER ─────────────────────────────────────── */}
        <section className="card">
          <p className="card-eyebrow">Ask the rocks anything</p>

          {/* Mic zone */}
          <div className="mic-zone">
            <span className="mic-zone-label">🎙 Voice</span>
            <AudioRecorder
              onAudioSubmitted={handleAudioSubmitted}
              disabled={isLoading}
            />
          </div>

          {/* OR TYPE divider */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            marginBottom: '0.85rem',
          }}>
            <div style={{ flex: 1, height: '1px', background: 'rgba(255,255,255,0.1)' }} />
            <span style={{
              fontSize: '0.58rem',
              fontWeight: 600,
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              color: 'rgba(253,248,238,0.3)',
            }}>
              or type
            </span>
            <div style={{ flex: 1, height: '1px', background: 'rgba(255,255,255,0.1)' }} />
          </div>

          {/* Text input row */}
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleTextQuery()}
              placeholder="Type your question…"
              className="query-input"
              disabled={isLoading}
              aria-label="Type your question"
            />
            <button
              onClick={() => handleTextQuery()}
              disabled={isLoading || !query.trim()}
              className="btn btn-ocean"
              aria-label="Submit question"
            >
              Ask
            </button>
          </div>
        </section>

        {/* ── EMPTY STATE ─────────────────────────────────────────── */}
        {showEmpty && (
          <div className="card fade-in" style={{ textAlign: 'center', padding: '1rem 1.25rem' }}>
            <p style={{
              fontSize: '0.58rem',
              fontWeight: 700,
              letterSpacing: '0.15em',
              textTransform: 'uppercase',
              color: 'rgba(253,248,238,0.4)',
              margin: '0 0 0.65rem',
            }}>
              Try asking
            </p>
            <div style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '0.45rem',
              justifyContent: 'center',
            }}>
              {EXAMPLES.map((q) => (
                <button
                  key={q}
                  onClick={() => handleExample(q)}
                  className="example-chip"
                  aria-label={`Ask: ${q}`}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ── PIPELINE INDICATOR ──────────────────────────────────── */}
        {showPipeline && (
          <div className="card fade-in" style={{ padding: '0.85rem 1.25rem' }}>
            <p style={{
              fontSize: '0.56rem',
              fontWeight: 700,
              letterSpacing: '0.14em',
              textTransform: 'uppercase',
              color: 'rgba(243,186,32,0.6)',
              margin: '0 0 0.65rem',
            }}>
              Pipeline
            </p>
            <PipelineIndicator stages={buildStages(pipelineMode, isVoice)} />
          </div>
        )}

        {/* ── LOADING ─────────────────────────────────────────────── */}
        {isLoading && (
          <div className="card fade-in" style={{ textAlign: 'center', padding: '1.5rem' }}>
            <LoadingSpinner mode={loadingMode} />
          </div>
        )}

        {/* ── ERROR ───────────────────────────────────────────────── */}
        {error && (
          <div
            className="card fade-in"
            style={{
              borderLeft: '3px solid #d94f4f',
              borderRadius: '0 14px 14px 0',
            }}
          >
            <p style={{
              fontSize: '0.62rem',
              fontWeight: 700,
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              color: '#e07070',
              marginBottom: '0.3rem',
            }}>
              Error
            </p>
            <p style={{ fontSize: '0.9rem', color: 'rgba(253,248,238,0.85)', margin: '0 0 0.6rem' }}>
              {error}
            </p>
            <button
              className="btn btn-ghost"
              style={{ fontSize: '0.78rem' }}
              onClick={() => { setError(''); setHasQueried(false) }}
            >
              Try again
            </button>
          </div>
        )}

        {/* ── ANSWER ──────────────────────────────────────────────── */}
        {hasResult && !isLoading && (
          <div className="fade-in">
            <AnswerDisplay
              query={query}
              answer={answer}
              evidence={evidence}
              confidence={confidence}
              grounded={grounded}
              language={language}
              latency={latency}
              sttLatency={sttLatency}
              verifying={verifying}
              unverifiedReason={unverifiedReason}
            />
          </div>
        )}

        {/* ── FOOTER ──────────────────────────────────────────────── */}
        <footer style={{
          textAlign: 'center',
          padding: '0.5rem 0 0',
          fontSize: '0.62rem',
          fontWeight: 600,
          letterSpacing: '0.12em',
          textTransform: 'uppercase',
          color: 'rgba(253,248,238,0.35)',
          textShadow: '0 1px 3px rgba(0,0,0,0.4)',
        }}>
          Hacker House Goa 2026 · Task 2 · RAG
        </footer>

      </div>
    </div>
  )
}