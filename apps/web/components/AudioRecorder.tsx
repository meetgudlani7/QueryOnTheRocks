'use client'

import { useState, useRef, useEffect } from 'react'

export interface AudioRecorderProps {
  onAudioSubmitted: (blob: Blob) => Promise<void>
}

// MediaRecorder doesn't support 'audio/wav' in any major browser — it
// records webm/opus (Chrome/Firefox) or mp4/aac (Safari). Picking the
// browser's real encoding (instead of just labeling the blob 'audio/wav')
// keeps the file's declared type honest end to end, since the backend
// forwards this content-type straight to Groq Whisper.
const PREFERRED_MIME_TYPES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg']

function getSupportedMimeType(): string | undefined {
  if (typeof MediaRecorder === 'undefined') return undefined
  return PREFERRED_MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type))
}

export default function AudioRecorder({ onAudioSubmitted }: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [audioUrl, setAudioUrl] = useState('')
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [error, setError] = useState('')
  const [isSupported, setIsSupported] = useState(false)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  useEffect(() => {
    // Check if audio recording is supported
    setIsSupported('MediaRecorder' in window)
  }, [])

  const startRecording = async () => {
    try {
      setError('')
      audioChunksRef.current = []
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mimeType = getSupportedMimeType()
      const mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data)
      }

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mediaRecorder.mimeType || 'audio/webm' })
        const audioUrl = URL.createObjectURL(audioBlob)
        setAudioBlob(audioBlob)
        setAudioUrl(audioUrl)
        stream.getTracks().forEach(track => track.stop())
      }
      
      mediaRecorder.start()
      setIsRecording(true)
    } catch (err) {
      setError('Could not access microphone. Please check permissions.')
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
    if (audioBlob) {
      await onAudioSubmitted(audioBlob)
      setAudioBlob(null)
      setAudioUrl('')
    }
  }

  const handleCancel = () => {
    setAudioBlob(null)
    setAudioUrl('')
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stream?.getTracks().forEach(track => track.stop())
    }
  }

  if (!isSupported) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <p className="text-yellow-700">
          Audio recording is not supported in your browser.
        </p>
      </div>
    )
  }

  return (
    <div className="recorder-container">
      {/* Recording Controls */}
      <div className="flex gap-4">
        {isRecording ? (
          <button
            onClick={stopRecording}
            className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Stop Recording
          </button>
        ) : (
          <button
            onClick={startRecording}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Start Recording
          </button>
        )}

        {audioBlob && !isRecording && (
          <>
            <button
              onClick={handleSubmit}
              className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
            >
              Submit
            </button>
            <button
              onClick={handleCancel}
              className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
            >
              Cancel
            </button>
          </>
        )}
      </div>

      {/* Error */}
      {error && (
        <p className="text-red-500 text-sm mt-2">{error}</p>
      )}

      {/* Audio Preview */}
      {audioUrl && (
        <div className="mt-4">
          <audio
            src={audioUrl}
            controls
            className="w-full"
          />
        </div>
      )}

      {/* Recording Indicator */}
      {isRecording && (
        <div className="mt-4 p-4 bg-blue-50 rounded-lg">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
            <p className="text-blue-700">Recording in progress...</p>
          </div>
        </div>
      )}
    </div>
  )
}
