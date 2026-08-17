'use client'

export default function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center gap-4">
      <div className="spinner" />
      <p className="text-gray-600">Processing your request...</p>
    </div>
  )
}
