// API Client utilities

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface QueryRequest {
  query: string
  language?: string
}

export interface QueryResponse {
  query: string
  answer: string
  evidence: string[]
  confidence: number
  grounded: boolean
  latency_ms: number
  language: string
  request_id: string
}

export interface AudioResponse {
  transcript: string
  language: string
  stt_latency_ms: number
  answer: string
  evidence: string[]
  confidence: number
  grounded: boolean
  rag_latency_ms: number
  request_id: string
}

// Query API
export async function queryAPI(request: QueryRequest): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/api/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    throw new Error('Query API failed')
  }

  return response.json()
}

// One event from the streaming query API (see apps/api/routes/query.py's
// _stream_query_events and pipeline/orchestrator.py::process_query_stream
// for the authoritative event shapes this mirrors).
export interface StreamEvent {
  type: 'token' | 'refused' | 'error' | 'unverified' | 'verified'
  text?: string
  reason?: string
  answer?: string
  evidence?: string[]
  evidence_ids?: string[]
  grounded?: boolean
  confidence?: number
  latency_ms?: number
  message?: string
  request_id?: string
}

// Streaming query API (roadmap Phase 21). Answer text arrives as a
// sequence of "token" events, immediately renderable; the answer is
// provisional until a terminal "verified" or "unverified" event closes
// the stream, since post-hoc groundedness validation can only run once
// generation has fully finished — see process_query_stream's docstring.
export async function queryAPIStream(request: QueryRequest, onEvent: (event: StreamEvent) => void): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/query?stream=true`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok || !response.body) {
    throw new Error('Streaming query API failed')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // Each backend frame is one blank-line-terminated "data: {json}"
    // block; split on the frame separator and hold back a possibly-
    // incomplete trailing frame for the next chunk rather than assuming
    // one network chunk always lines up with one SSE frame.
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      const line = frame.trim()
      if (!line.startsWith('data: ')) continue
      const payload = line.slice('data: '.length)
      if (payload === '[DONE]') return
      try {
        onEvent(JSON.parse(payload) as StreamEvent)
      } catch {
        // A single malformed frame shouldn't abort an otherwise-good stream.
      }
    }
  }
}

// Audio API
export async function uploadAudio(audioBlob: Blob): Promise<AudioResponse> {
  const formData = new FormData()
  // Filename extension mirrors the blob's actual MIME type (set by
  // AudioRecorder from what MediaRecorder really encoded) so the backend's
  // content-type and the file extension it hands to Groq Whisper agree.
  const extension = audioBlob.type.split('/')[1]?.split(';')[0] || 'webm'
  formData.append('file', audioBlob, `recording.${extension}`)

  const response = await fetch(`${API_BASE_URL}/api/audio`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error('Audio upload failed')
  }

  return response.json()
}

// Health check
export async function healthCheck(): Promise<boolean> {
  const response = await fetch(`${API_BASE_URL}/health`)
  return response.ok
}
