'use client'

import { useState } from 'react'
import AudioRecorder from '../components/AudioRecorder'
import AnswerDisplay from '../components/AnswerDisplay'
import LoadingSpinner from '../components/LoadingSpinner'
import { queryAPI, uploadAudio } from '../lib/api'

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
  const [error, setError] = useState('')

  const handleTextQuery = async () => {
    if (!query.trim()) return

    setIsLoading(true)
    setError('')
    setSttLatency(undefined)

    try {
      const data = await queryAPI({ query, language: 'en' })
      setAnswer(data.answer)
      setEvidence(data.evidence || [])
      setConfidence(data.confidence || 0)
      setGrounded(data.grounded || false)
      setLanguage(data.language || 'en')
      setLatency(data.latency_ms || 0)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setIsLoading(false)
    }
  }

  const handleAudioSubmitted = async (audioBlob: Blob) => {
    setIsLoading(true)
    setError('')
    setQuery('')
    setAnswer('')
    setEvidence([])

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

  return (
    <main className="min-h-screen bg-gradient-to-b from-blue-50 to-white p-8">
      <div className="container">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Voice-Enabled RAG System
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Ask questions via voice or text and get grounded answers from our knowledge base
          </p>
        </header>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto">
          {/* Recorder Section */}
          <section className="bg-white rounded-2xl shadow-lg p-8 mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">
              Voice Input
            </h2>
            <AudioRecorder onAudioSubmitted={handleAudioSubmitted} />
          </section>

          {/* Text Input Section */}
          <section className="bg-white rounded-2xl shadow-lg p-8 mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">
              Text Input
            </h2>
            <div className="flex gap-4">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a question..."
                className="flex-1 px-4 py-3 border border-gray-300 rounded-lg bg-white text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={isLoading}
                onKeyPress={(e) => e.key === 'Enter' && handleTextQuery()}
              />
              <button
                onClick={handleTextQuery}
                disabled={isLoading || !query.trim()}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors"
              >
                Ask
              </button>
            </div>
          </section>

          {/* Loading State */}
          {isLoading && (
            <div className="flex justify-center py-8">
              <LoadingSpinner />
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-8">
              <p className="text-red-600">{error}</p>
            </div>
          )}

          {/* Answer Display */}
          {(answer || evidence.length > 0) && !isLoading && (
            <AnswerDisplay
              query={query}
              answer={answer}
              evidence={evidence}
              confidence={confidence}
              grounded={grounded}
              language={language}
              latency={latency}
              sttLatency={sttLatency}
            />
          )}
        </div>

        {/* Footer */}
        <footer className="text-center text-gray-500 text-sm mt-12">
          <p>HH Goa 2026 - Task 2: Voice-Enabled RAG System</p>
        </footer>
      </div>
    </main>
  )
}
