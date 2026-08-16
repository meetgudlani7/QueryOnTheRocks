'use client'

import { useState, useRef, useEffect } from 'react'

export interface AudioRecorderProps {
  onAudioSubmitted: (blob: Blob) => Promise<void>
  disabled?: boolean
}

// MediaRecorder doesn't support 'audio/wav' in any major browser — it
// records webm/opus (Chrome/Firefox) or mp4/aac (Safari). Picking the
// browser's real encoding keeps the file's declared type honest end to
// end, since the backend forwards this content-type straight to Groq Whisper.
const PREFERRED_MIME_TYPES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/mp4',
  'audio/ogg',
]

function getSupportedMimeType(): string | undefined {
  if (typeof MediaRecorder === 'undefined') return undefined
  return PREFERRED_MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type))
}

// Mic icon SVG
function MicIcon({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
      aria-hidden="true">
      <rect x="9" y="2" width="6" height="11" rx="3" />
      <path d="M5 10a7 7 0 0 0 14 0" />
      <line x1="12" y1="19" x2="12" y2="22" />
      <line x1="9" y1="22" x2="15" y2="22" />
    </svg>
  )
}

// Stop icon SVG
function StopIcon({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor"
      aria-hidden="true">
      <rect x="5" y="5" width="14" height="14" rx="2" />
    </svg>
  )
}

// Send icon SVG
function SendIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
      aria-hidden="true">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  )
}

export default function AudioRecorder({ onAudioSubmitted, disabled = false }: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [audioUrl, setAudioUrl] = useState('')
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [error, setError] = useState('')
  const [isSupported, setIsSupported] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [elapsed, setElapsed] = useState(0)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    setIsSupported('MediaRecorder' in window)
  }, [])

  // Recording timer
  useEffect(() => {
    if (isRecording) {
      setElapsed(0)
      timerRef.current = setInterval(() => {
        setElapsed((s) => s + 1)
      }, 1000)
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [isRecording])

  const formatElapsed = (s: number) => {
    const m = Math.floor(s / 60).toString().padStart(2, '0')
    const sec = (s % 60).toString().padStart(2, '0')
    return `${m}:${sec}`
  }

  const startRecording = async () => {
    try {
      setError('')
      setAudioBlob(null)
      setAudioUrl('')
      audioChunksRef.current = []

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mimeType = getSupportedMimeType()
      const mediaRecorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data)
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, {
          type: mediaRecorder.mimeType || 'audio/webm',
        })
        const url = URL.createObjectURL(blob)
        setAudioBlob(blob)
        setAudioUrl(url)
        stream.getTracks().forEach((track) => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
    } catch (err) {
      if (err instanceof Error && err.name === 'NotAllowedError') {
        setError('Microphone access denied. Please allow microphone permissions and try again.')
      } else {
        setError('Could not access microphone. Please check your device settings.')
      }
      console.error('Recording error:', err)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const handleSubmit = async () => {
    if (!audioBlob) return
    setIsSubmitting(true)
    try {
      await onAudioSubmitted(audioBlob)
    } finally {
      setIsSubmitting(false)
      setAudioBlob(null)
      setAudioUrl('')
      setElapsed(0)
    }
  }

  const handleCancel = () => {
    setAudioBlob(null)
    setAudioUrl('')
    setElapsed(0)
    setError('')
    mediaRecorderRef.current?.stream
      ?.getTracks()
      .forEach((track) => track.stop())
  }

  // ── Browser not supported ──────────────────────────────────────────
  if (!isSupported) {
    return (
      <div style={{
        padding: '0.6rem 0.9rem',
        background: 'rgba(122,30,30,0.08)',
        borderRadius: '8px',
        fontSize: '0.82rem',
        color: 'var(--error-fg)',
      }}>
        Audio recording is not supported in this browser.
      </div>
    )
  }

  // ── IDLE: no recording, no blob ────────────────────────────────────
  if (!isRecording && !audioBlob) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
        <button
          onClick={startRecording}
          disabled={disabled}
          className={`mic-btn${disabled ? '' : ''}`}
          aria-label="Start voice recording"
          title="Click to speak"
        >
          <MicIcon />
        </button>
        <span style={{
          fontSize: '0.68rem',
          fontWeight: 600,
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
          color: 'rgba(44,44,44,0.5)',
        }}>
          Speak
        </span>
        {error && (
          <p style={{ fontSize: '0.8rem', color: 'var(--error-fg)', textAlign: 'center', maxWidth: '260px' }}>
            {error}
          </p>
        )}
      </div>
    )
  }

  // ── RECORDING ─────────────────────────────────────────────────────
  if (isRecording) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
        <button
          onClick={stopRecording}
          className="mic-btn recording"
          aria-label="Stop recording"
          title="Click to stop"
        >
          <StopIcon />
        </button>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.15rem' }}>
          <span style={{
            fontSize: '0.68rem',
            fontWeight: 700,
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            color: '#7A1E1E',
          }}>
            Listening…
          </span>
          <span style={{
            fontSize: '0.72rem',
            fontWeight: 600,
            color: 'rgba(44,44,44,0.5)',
            fontVariantNumeric: 'tabular-nums',
          }}>
            {formatElapsed(elapsed)}
          </span>
        </div>
      </div>
    )
  }

  // ── RECORDED: blob ready, waiting for submit or cancel ─────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem', width: '100%' }}>
      {/* Audio preview */}
      <audio
        src={audioUrl}
        controls
        style={{ width: '100%', height: '36px', accentColor: 'var(--ocean)' }}
        aria-label="Recording preview"
      />

      {/* Action row */}
      <div style={{ display: 'flex', gap: '0.5rem' }}>
        <button
          onClick={handleSubmit}
          disabled={isSubmitting}
          className="btn btn-ocean"
          aria-label="Submit recording"
        >
          <SendIcon />
          {isSubmitting ? 'Sending…' : 'Send recording'}
        </button>
        <button
          onClick={handleCancel}
          disabled={isSubmitting}
          className="btn btn-ghost"
          aria-label="Cancel recording"
        >
          Discard
        </button>
      </div>

      {error && (
        <p style={{ fontSize: '0.8rem', color: 'var(--error-fg)' }}>{error}</p>
      )}
    </div>
  )
}