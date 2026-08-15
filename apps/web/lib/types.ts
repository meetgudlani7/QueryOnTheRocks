// Type definitions for the application

export interface Query {
  id?: number
  text: string
  language: string
}

export interface Answer {
  query: string
  text: string
  evidence: string[]
  confidence: number
  latencyMs: number
}

export interface AudioData {
  blob: Blob
  url: string
}

export interface Metrics {
  confidence: number
  latencyMs: number
  evidenceCount: number
}
