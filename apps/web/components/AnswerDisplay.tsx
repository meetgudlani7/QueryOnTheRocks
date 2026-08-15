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
}

export default function AnswerDisplay({
  query,
  answer,
  evidence,
  confidence,
  grounded,
  language,
  latency,
  sttLatency,
}: AnswerDisplayProps) {
  const [expanded, setExpanded] = useState(false)

  const formatLatency = (ms: number) => {
    if (ms < 1000) {
      return `${ms.toFixed(1)}ms`
    }
    return `${(ms / 1000).toFixed(2)}s`
  }

  const formatConfidence = (conf: number) => {
    return `${(conf * 100).toFixed(1)}%`
  }

  return (
    <section className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl shadow-lg p-8">
      <h2 className="text-2xl font-semibold text-gray-800 mb-6">
        Answer
      </h2>

      {/* Query */}
      {query && (
        <div className="mb-6 p-4 bg-white rounded-lg shadow-sm">
          <p className="text-gray-700">
            <span className="font-medium">Question:</span> {query}
          </p>
        </div>
      )}

      {/* Answer */}
      <div className="answer-card mb-6">
        <p className="text-xl text-gray-800 leading-relaxed">
          {answer || 'No answer available'}
        </p>
      </div>

      {/* Metrics — mirrors the guide's "show the engineering" panel */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-6">
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <p className="text-sm text-gray-500 uppercase tracking-wider mb-1">Language</p>
          <p className="text-lg font-semibold text-gray-800">{language || 'unknown'}</p>
        </div>
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <p className="text-sm text-gray-500 uppercase tracking-wider mb-1">Retrieval</p>
          <p className="text-lg font-semibold text-gray-800">Hybrid</p>
        </div>
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <p className="text-sm text-gray-500 uppercase tracking-wider mb-1">Grounded</p>
          <p className={`text-lg font-semibold ${grounded ? 'text-green-600' : 'text-red-600'}`}>
            {grounded ? 'Yes' : 'No'}
          </p>
        </div>
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <p className="text-sm text-gray-500 uppercase tracking-wider mb-1">Confidence</p>
          <p className="text-2xl font-semibold text-blue-600">{formatConfidence(confidence)}</p>
        </div>
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <p className="text-sm text-gray-500 uppercase tracking-wider mb-1">RAG Latency</p>
          <p className="text-2xl font-semibold text-green-600">{formatLatency(latency)}</p>
        </div>
        {sttLatency !== undefined && (
          <div className="bg-white rounded-lg p-4 shadow-sm">
            <p className="text-sm text-gray-500 uppercase tracking-wider mb-1">STT Latency</p>
            <p className="text-2xl font-semibold text-green-600">{formatLatency(sttLatency)}</p>
          </div>
        )}
      </div>

      {/* Evidence */}
      {evidence.length > 0 && (
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center justify-between w-full text-left"
          >
            <h3 className="text-lg font-semibold text-gray-800">
              Evidence ({evidence.length})
            </h3>
            <svg
              className={`w-5 h-5 text-gray-500 transition-transform ${expanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {expanded && (
            <div className="mt-4 space-y-4">
              {evidence.map((item, index) => (
                <div
                  key={index}
                  className="p-4 bg-gray-50 rounded-lg border border-gray-200"
                >
                  <p className="text-gray-700 text-sm">{item}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  )
}
