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
