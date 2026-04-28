/**
 * ChatWidget.jsx
 *
 * Floating domain-restricted AI chat assistant (bottom-right of screen).
 * Posts user messages to /api/v1/chat and renders the conversation in-place
 * without triggering a full page reload.
 */

import { useState, useRef, useEffect } from 'react'

const API_BASE = import.meta.env.VITE_API_URL ?? ''

export default function ChatWidget() {
  const [open, setOpen] = useState(false)
  const [history, setHistory] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  // Auto-scroll to the latest message whenever history changes
  useEffect(() => {
    if (open) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [history, open])

  const sendMessage = async () => {
    const trimmed = input.trim()
    if (!trimmed || loading) return

    const userEntry = { id: Date.now(), role: 'user', text: trimmed }
    setHistory((prev) => [...prev, userEntry])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(`${API_BASE}/api/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_message: trimmed }),
      })

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}))
        throw new Error(detail?.detail ?? `Server error ${res.status}`)
      }

      const data = await res.json()
      setHistory((prev) => [...prev, { id: Date.now() + 1, role: 'assistant', text: data.response }])
    } catch (err) {
      setHistory((prev) => [
        ...prev,
        { id: Date.now() + 1, role: 'error', text: err.message ?? 'An unexpected error occurred.' },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-3">
      {/* Chat panel */}
      {open && (
        <div
          className="w-80 sm:w-96 flex flex-col rounded-2xl shadow-2xl border
            border-slate-200 dark:border-slate-700
            bg-white dark:bg-slate-800 overflow-hidden"
          style={{ height: '28rem' }}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3
            bg-indigo-600 text-white">
            <div className="flex items-center gap-2">
              <span className="text-lg">💬</span>
              <div>
                <p className="text-sm font-semibold leading-none">Financial Assistant</p>
                <p className="text-xs text-indigo-200 mt-0.5">Loan Eligibility Advisor</p>
              </div>
            </div>
            <button
              onClick={() => setOpen(false)}
              aria-label="Close chat"
              className="text-indigo-200 hover:text-white transition-colors text-lg leading-none"
            >
              ✕
            </button>
          </div>

          {/* Message history */}
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 text-sm">
            {history.length === 0 && (
              <p className="text-slate-400 dark:text-slate-500 text-center mt-6 text-xs leading-relaxed">
                Ask me about loan parameters, credit scores, DTI, or your eligibility results.
              </p>
            )}
            {history.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-xl px-3 py-2 leading-relaxed whitespace-pre-wrap break-words
                    ${msg.role === 'user'
                      ? 'bg-indigo-600 text-white rounded-br-none'
                      : msg.role === 'error'
                      ? 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300 rounded-bl-none'
                      : 'bg-slate-100 dark:bg-slate-700 text-slate-800 dark:text-slate-100 rounded-bl-none'
                    }`}
                >
                  {msg.text}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-slate-100 dark:bg-slate-700 rounded-xl rounded-bl-none px-4 py-2">
                  <span className="inline-flex gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:0ms]" />
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:150ms]" />
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:300ms]" />
                  </span>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input row */}
          <div className="px-3 pb-3 pt-2 border-t border-slate-200 dark:border-slate-700 flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
              placeholder="Ask about loans, DTI, credit…"
              className="flex-1 rounded-lg border border-slate-300 dark:border-slate-600
                bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100
                text-sm px-3 py-2 outline-none
                focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500
                disabled:opacity-50 placeholder-slate-400"
            />
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              aria-label="Send message"
              className="rounded-lg bg-indigo-600 hover:bg-indigo-700
                text-white px-3 py-2 text-sm font-medium transition-colors
                disabled:opacity-40 disabled:cursor-not-allowed"
            >
              ➤
            </button>
          </div>
        </div>
      )}

      {/* Toggle FAB */}
      <button
        onClick={() => setOpen((prev) => !prev)}
        aria-label={open ? 'Close financial assistant' : 'Open financial assistant'}
        className="w-14 h-14 rounded-full bg-indigo-600 hover:bg-indigo-700
          text-white text-2xl shadow-xl flex items-center justify-center
          transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
      >
        {open ? '✕' : '💬'}
      </button>
    </div>
  )
}
