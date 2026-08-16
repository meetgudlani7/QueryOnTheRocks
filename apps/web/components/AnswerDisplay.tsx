'use client'

import { useState } from 'react'

export interface AnswerDisplayProps {
  query: string
  answer: string
  evidence: string[]
  confidence: number
  grounded: boolean
  language: string
  latency: number
  sttLatency?: number
  verifying?: boolean
  unverifiedReason?: string
}

// ── Small primitives ────────────────────────────────────────────────────

function GroundedBadge({ grounded, verifying }: { grounded: boolean; verifying?: boolean }) {
  if (verifying) {
    return (
      <span className="badge badge-verifying" aria-live="polite">
        <span style={{
          width: '6px', height: '6px', borderRadius: '50%',
          background: 'var(--ocean)', display: 'inline-block',
          animation: 'mic-pulse 1.2s ease-in-out infinite',
        }} />
        Verifying…
      </span>
    )
  }
  if (grounded) {
    return (
      <span className="badge badge-grounded" aria-label="Answer is grounded">
        ✓ Grounded
      </span>
    )
  }
  return (
    <span className="badge badge-ungrounded" aria-label="Answer is not grounded">
      ! Not grounded
    </span>
  )
}

function LatencyChip({ ms, label }: { ms: number; label: string }) {
  const target = 200
  const isFast = ms > 0 && ms <= target
  const isSlow = ms > target

  return (
    <div className="metric-chip">
      <span className="label">{label}</span>
      <span
        className={`metric-value ${isFast ? 'fast' : isSlow ? 'slow' : ''}`}
        aria-label={`${label}: ${ms.toFixed(1)} milliseconds`}
      >
        ⚡ {ms.toFixed(1)} ms
      </span>
      {label === 'RAG Latency' && ms > 0 && (
        <span style={{
          fontSize: '0.62rem',
          color: isFast ? 'var(--olive)' : 'var(--burnt)',
          fontWeight: 600,
        }}>
          {isFast ? `${(target - ms).toFixed(0)} ms under target` : `target < ${target} ms`}
        </span>
      )}
    </div>
  )
}

function ConfidenceChip({ confidence }: { confidence: number }) {
  const pct = (confidence * 100).toFixed(1)
  return (
    <div className="metric-chip">
      <span className="label">Confidence</span>
      <span className="metric-value" style={{ color: 'var(--ocean)' }}>
        {pct}%
      </span>
    </div>
  )
}

function Divider() {
  return (
    <div style={{
      height: '1px',
      background: 'var(--border-warm)',
      margin: '0.85rem 0',
    }} />
  )
}

// ── Main component ──────────────────────────────────────────────────────

export default function AnswerDisplay({
  query,
  answer,
  evidence,
  confidence,
  grounded,
  language,
  latency,
  sttLatency,
  verifying = false,
  unverifiedReason,
}: AnswerDisplayProps) {
  const [evidenceOpen, setEvidenceOpen] = useState(false)

  const showMetrics = latency > 0 || confidence > 0 || sttLatency !== undefined

  return (
    <section className="card-solid fade-in" aria-label="Query result">

      {/* ── TRANSCRIPT ───────────────────────────────────────────── */}
      {query && (
        <>
          <div>
            <p className="label" style={{ marginBottom: '0.3rem' }}>You asked</p>
            <p style={{
              fontSize: '0.9rem',
              color: 'var(--charcoal)',
              fontStyle: 'italic',
              lineHeight: 1.5,
              margin: 0,
            }}>
              "{query}"
            </p>
          </div>
          <Divider />
        </>
      )}

      {/* ── ANSWER ───────────────────────────────────────────────── */}
      <div>
        <p className="label" style={{ marginBottom: '0.5rem' }}>Answer</p>
        <div className="answer-card">
          <p className="answer-text" style={{ margin: 0 }}>
            {answer || 'No answer available.'}
            {verifying && (
              <span
                className="blink"
                style={{
                  display: 'inline-block',
                  width: '2px',
                  height: '1.1em',
                  background: 'var(--ocean)',
                  marginLeft: '3px',
                  verticalAlign: 'text-bottom',
                }}
                aria-hidden="true"
              />
            )}
          </p>
        </div>
      </div>

      {/* ── UNVERIFIED WARNING ───────────────────────────────────── */}
      {!verifying && unverifiedReason && (
        <>
          <Divider />
          <div style={{
            background: 'rgba(191,100,21,0.08)',
            border: '1px solid rgba(191,100,21,0.25)',
            borderRadius: '8px',
            padding: '0.65rem 0.9rem',
          }}>
            <p style={{
              fontSize: '0.78rem',
              fontWeight: 700,
              color: 'var(--burnt)',
              margin: '0 0 0.2rem',
            }}>
              Not fully verified
            </p>
            <p style={{
              fontSize: '0.78rem',
              color: 'var(--charcoal)',
              margin: 0,
              lineHeight: 1.5,
            }}>
              {unverifiedReason}
            </p>
          </div>
        </>
      )}

      <Divider />

      {/* ── STATUS ROW ───────────────────────────────────────────── */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '0.6rem',
      }}>
        <GroundedBadge grounded={grounded} verifying={verifying} />

        {language && (
          <span className="badge" style={{
            background: 'rgba(75,158,191,0.1)',
            color: 'var(--ocean)',
            border: '1px solid rgba(75,158,191,0.25)',
          }}>
            {language.toUpperCase()}
          </span>
        )}

        <span className="badge" style={{
          background: 'rgba(82,100,59,0.08)',
          color: 'var(--olive)',
          border: '1px solid rgba(82,100,59,0.2)',
        }}>
          Hybrid retrieval
        </span>
      </div>

      {/* ── METRICS ──────────────────────────────────────────────── */}
      {showMetrics && (
        <>
          <Divider />
          <div style={{
            display: 'flex',
            gap: '1.5rem',
            flexWrap: 'wrap',
          }}>
            {latency > 0 && (
              <LatencyChip ms={latency} label="RAG Latency" />
            )}
            {sttLatency !== undefined && sttLatency > 0 && (
              <LatencyChip ms={sttLatency} label="STT Latency" />
            )}
            {confidence > 0 && (
              <ConfidenceChip confidence={confidence} />
            )}
          </div>
        </>
      )}

      {/* ── EVIDENCE ─────────────────────────────────────────────── */}
      {evidence.length > 0 && (
        <>
          <Divider />
          <div>
            <button
              onClick={() => setEvidenceOpen((o) => !o)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                padding: 0,
                width: '100%',
                textAlign: 'left',
              }}
              aria-expanded={evidenceOpen}
              aria-controls="evidence-list"
            >
              <span className="label" style={{ color: 'var(--ocean)' }}>
                Retrieved evidence
              </span>
              <span style={{
                fontSize: '0.65rem',
                fontWeight: 700,
                color: 'rgba(75,158,191,0.7)',
                background: 'rgba(75,158,191,0.1)',
                borderRadius: '20px',
                padding: '0.1rem 0.45rem',
              }}>
                {evidence.length}
              </span>
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="var(--ocean)"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{
                  marginLeft: 'auto',
                  transform: evidenceOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                  transition: 'transform 0.2s',
                }}
                aria-hidden="true"
              >
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>

            {evidenceOpen && (
              <div
                id="evidence-list"
                className="fade-in"
                style={{
                  marginTop: '0.75rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.5rem',
                }}
              >
                {evidence.map((item, index) => (
                  <div key={index} className="evidence-item">
                    <span style={{
                      fontSize: '0.6rem',
                      fontWeight: 700,
                      letterSpacing: '0.08em',
                      textTransform: 'uppercase',
                      color: 'var(--burnt)',
                      display: 'block',
                      marginBottom: '0.2rem',
                    }}>
                      Source {index + 1}
                    </span>
                    {item}
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}

    </section>
  )
}