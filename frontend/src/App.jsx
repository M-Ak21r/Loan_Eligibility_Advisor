/**
 * App.jsx
 *
 * Root component for the Loan Eligibility Advisor.
 *
 * State:
 *   - result      : latest prediction response from the API (or null)
 *   - loading     : true while awaiting the API response
 *   - error       : error message string (or null)
 *
 * On mount (useEffect) the component verifies backend reachability via /health.
 */

import { useState, useEffect } from 'react'
import LoanForm from './components/LoanForm'
import ResultCard from './components/ResultCard'
import ChatWidget from './components/ChatWidget'

const API_BASE = import.meta.env.VITE_API_URL ?? ''

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [backendReady, setBackendReady] = useState(null) // null = checking

  // ── Lifecycle: verify backend on mount ────────────────────────────────────
  useEffect(() => {
    const controller = new AbortController()
    fetch(`${API_BASE}/health`, { signal: controller.signal })
      .then((res) => setBackendReady(res.ok))
      .catch(() => setBackendReady(false))
    return () => controller.abort()
  }, [])

  // ── Async prediction request ───────────────────────────────────────────────
  const handleSubmit = async (payload) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`${API_BASE}/api/v1/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail?.detail ?? `Server error ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message ?? 'An unexpected error occurred.')
      }
    } finally {
      setLoading(false)
    }
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-indigo-50 to-slate-100
      dark:from-slate-900 dark:via-slate-800 dark:to-slate-900
      flex flex-col items-center justify-start px-4 py-12 font-sans">

      {/* Header */}
      <header className="mb-10 text-center max-w-2xl">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl
          bg-indigo-600 text-white text-3xl mb-4 shadow-lg">
          🏦
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-slate-900
          dark:text-slate-100">
          Loan Eligibility <span className="text-indigo-600">Advisor</span>
        </h1>
        <p className="mt-3 text-base sm:text-lg text-slate-500 dark:text-slate-400">
          Enter your financial details to receive an instant AI-powered loan
          eligibility assessment.
        </p>

        {/* Backend status pill */}
        {backendReady === false && (
          <div className="mt-4 inline-flex items-center gap-2 rounded-full
            bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300
            px-4 py-1.5 text-sm font-medium">
            <span className="inline-block w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            Backend unavailable — start the FastAPI server to enable predictions
          </div>
        )}
        {backendReady === true && (
          <div className="mt-4 inline-flex items-center gap-2 rounded-full
            bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300
            px-4 py-1.5 text-sm font-medium">
            <span className="inline-block w-2 h-2 rounded-full bg-emerald-500" />
            Backend connected
          </div>
        )}
      </header>

      {/* Main card */}
      <main className="w-full max-w-2xl space-y-6">
        <div className="rounded-2xl bg-white dark:bg-slate-800 shadow-xl p-6 sm:p-8 border
          border-slate-200 dark:border-slate-700">
          <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-5">
            Application Details
          </h2>
          <LoanForm onSubmit={handleSubmit} loading={loading} />
        </div>

        {/* Error banner */}
        {error && (
          <div className="rounded-xl border border-red-300 dark:border-red-700
            bg-red-50 dark:bg-red-900/30 px-5 py-4 text-sm text-red-700 dark:text-red-300">
            <span className="font-semibold">Error: </span>{error}
          </div>
        )}

        {/* Result card */}
        <ResultCard result={result} />
      </main>

      {/* Footer */}
      <footer className="mt-16 text-xs text-slate-400 dark:text-slate-600 text-center">
        Loan Eligibility Advisor MVP · Powered by RandomForest &amp; FastAPI
      </footer>

      {/* Floating chat widget */}
      <ChatWidget />
    </div>
  )
}
