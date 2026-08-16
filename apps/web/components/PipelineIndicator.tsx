'use client'

type StageStatus = 'inactive' | 'active' | 'complete' | 'error'

interface Stage {
  key: string
  label: string
  status: StageStatus
}

interface PipelineIndicatorProps {
  stages: Stage[]
}

function StageDot({ status }: { status: StageStatus }) {
  const base: React.CSSProperties = {
    width: '9px',
    height: '9px',
    borderRadius: '50%',
    transition: 'background 0.3s, box-shadow 0.3s',
    flexShrink: 0,
  }

  const styles: Record<StageStatus, React.CSSProperties> = {
    inactive: { background: 'rgba(44,44,44,0.18)' },
    active: {
      background: 'var(--mustard)',
      boxShadow: '0 0 0 3px rgba(243,186,32,0.25)',
    },
    complete: { background: 'var(--olive)' },
    error: {
      background: '#7A1E1E',
      boxShadow: '0 0 0 3px rgba(122,30,30,0.2)',
    },
  }

  return <div style={{ ...base, ...styles[status] }} aria-hidden="true" />
}

function StageLabel({ label, status }: { label: string; status: StageStatus }) {
  const colors: Record<StageStatus, string> = {
    inactive: 'rgba(44,44,44,0.35)',
    active: 'var(--mustard)',
    complete: 'var(--olive)',
    error: '#7A1E1E',
  }

  return (
    <span style={{
      fontSize: '0.55rem',
      fontWeight: 700,
      letterSpacing: '0.1em',
      textTransform: 'uppercase',
      color: colors[status],
      transition: 'color 0.3s',
      whiteSpace: 'nowrap',
    }}>
      {label}
    </span>
  )
}

function ConnectorLine({ status }: { status: 'inactive' | 'complete' }) {
  return (
    <div style={{
      flex: 1,
      height: '1px',
      background: status === 'complete'
        ? 'var(--olive)'
        : 'rgba(44,44,44,0.15)',
      transition: 'background 0.3s',
      marginBottom: '1rem', // aligns with dot row
    }} aria-hidden="true" />
  )
}

export default function PipelineIndicator({ stages }: PipelineIndicatorProps) {
  return (
    <div
      role="status"
      aria-label="Pipeline progress"
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        width: '100%',
        padding: '0.1rem 0',
      }}
    >
      {stages.map((stage, i) => (
        <div key={stage.key} style={{ display: 'contents' }}>
          {/* Stage column */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '0.25rem',
          }}>
            <StageDot status={stage.status} />
            <StageLabel label={stage.label} status={stage.status} />
          </div>

          {/* Connector between stages */}
          {i < stages.length - 1 && (
            <ConnectorLine
              status={stage.status === 'complete' ? 'complete' : 'inactive'}
            />
          )}
        </div>
      ))}
    </div>
  )
}

// ── Helper: derive stage list from page state ──────────────────────────
export type PipelineMode = 'idle' | 'stt' | 'loading' | 'verifying' | 'complete' | 'error'

export function buildStages(mode: PipelineMode, isVoice: boolean): Stage[] {
  const s = (status: StageStatus): StageStatus => status

  switch (mode) {
    case 'idle':
      return [
        { key: 'voice',      label: 'Voice',      status: s('inactive') },
        { key: 'transcribe', label: 'Transcribe',  status: s('inactive') },
        { key: 'retrieve',   label: 'Retrieve',    status: s('inactive') },
        { key: 'generate',   label: 'Generate',    status: s('inactive') },
        { key: 'verify',     label: 'Verify',      status: s('inactive') },
      ]

    case 'stt':
      // Voice uploaded, STT in progress
      return [
        { key: 'voice',      label: 'Voice',      status: s('complete') },
        { key: 'transcribe', label: 'Transcribe',  status: s('active')  },
        { key: 'retrieve',   label: 'Retrieve',    status: s('inactive') },
        { key: 'generate',   label: 'Generate',    status: s('inactive') },
        { key: 'verify',     label: 'Verify',      status: s('inactive') },
      ]

    case 'loading':
      // Text submitted OR post-STT RAG running
      return [
        { key: 'voice',      label: 'Voice',      status: isVoice ? s('complete') : s('inactive') },
        { key: 'transcribe', label: 'Transcribe',  status: isVoice ? s('complete') : s('inactive') },
        { key: 'retrieve',   label: 'Retrieve',    status: s('active')  },
        { key: 'generate',   label: 'Generate',    status: s('inactive') },
        { key: 'verify',     label: 'Verify',      status: s('inactive') },
      ]

    case 'verifying':
      // Tokens streaming in, post-hoc groundedness check pending
      return [
        { key: 'voice',      label: 'Voice',      status: isVoice ? s('complete') : s('inactive') },
        { key: 'transcribe', label: 'Transcribe',  status: isVoice ? s('complete') : s('inactive') },
        { key: 'retrieve',   label: 'Retrieve',    status: s('complete') },
        { key: 'generate',   label: 'Generate',    status: s('complete') },
        { key: 'verify',     label: 'Verify',      status: s('active')  },
      ]

    case 'complete':
      return [
        { key: 'voice',      label: 'Voice',      status: isVoice ? s('complete') : s('inactive') },
        { key: 'transcribe', label: 'Transcribe',  status: isVoice ? s('complete') : s('inactive') },
        { key: 'retrieve',   label: 'Retrieve',    status: s('complete') },
        { key: 'generate',   label: 'Generate',    status: s('complete') },
        { key: 'verify',     label: 'Verify',      status: s('complete') },
      ]

    case 'error':
      return [
        { key: 'voice',      label: 'Voice',      status: isVoice ? s('complete') : s('inactive') },
        { key: 'transcribe', label: 'Transcribe',  status: isVoice ? s('complete') : s('inactive') },
        { key: 'retrieve',   label: 'Retrieve',    status: s('error')   },
        { key: 'generate',   label: 'Generate',    status: s('inactive') },
        { key: 'verify',     label: 'Verify',      status: s('inactive') },
      ]
  }
}