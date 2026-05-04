/**
 * ChatWidget.jsx
 *
 * Floating financial chat assistant for loan-eligibility questions.
 */

import { useEffect, useRef, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL ?? ''

export default function ChatWidget({ application, prediction }) {
  const [open, setOpen] = useState(false)
  const [history, setHistory] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    if (open) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [history, open])

  const sendMessage = async () => {
    const trimmed = input.trim()
    if (!trimmed || loading) return

    const userEntry = { id: crypto.randomUUID(), role: 'user', text: trimmed }
    setHistory((prev) => [...prev, userEntry])
    setInput('')
    setLoading(true)

    try {
      const response = await fetch(`${API_BASE}/api/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_message: trimmed,
          application,
          prediction,
        }),
      })

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail?.detail ?? `Server error ${response.status}`)
      }

      const data = await response.json()
      setHistory((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: 'assistant', text: data.response },
      ])
    } catch (err) {
      setHistory((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'error',
          text: err.message ?? 'An unexpected error occurred.',
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-3">
      {open && (
        <section
          className="flex h-[28rem] w-[min(calc(100vw-2rem),24rem)] flex-col overflow-hidden
            rounded-2xl border border-slate-200 bg-white shadow-2xl
            dark:border-slate-700 dark:bg-slate-800"
          aria-label="Financial assistant chat"
        >
          <header className="flex items-center justify-between bg-indigo-600 px-4 py-3 text-white">
            <div>
              <p className="text-sm font-semibold leading-none">Financial Assistant</p>
              <p className="mt-1 text-xs text-indigo-100">Loan Eligibility Advisor</p>
            </div>
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close chat"
              className="grid h-8 w-8 place-items-center rounded-lg text-lg leading-none
                text-indigo-100 transition hover:bg-indigo-500 hover:text-white"
            >
              x
            </button>
          </header>

          <div className="flex-1 space-y-3 overflow-y-auto px-4 py-3 text-sm">
            {history.length === 0 && (
              <p className="mt-6 text-center text-xs leading-relaxed text-slate-400 dark:text-slate-500">
                Ask about your eligibility result, loan burden, income, default history, or next steps.
              </p>
            )}

            {history.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <p
                  className={`max-w-[82%] whitespace-pre-wrap break-words rounded-xl px-3 py-2 leading-relaxed
                    ${message.role === 'user'
                      ? 'rounded-br-none bg-indigo-600 text-white'
                      : message.role === 'error'
                        ? 'rounded-bl-none bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300'
                        : 'rounded-bl-none bg-slate-100 text-slate-800 dark:bg-slate-700 dark:text-slate-100'
                    }`}
                >
                  {message.text}
                </p>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <p className="rounded-xl rounded-bl-none bg-slate-100 px-3 py-2 text-slate-500 dark:bg-slate-700 dark:text-slate-300">
                  Thinking...
                </p>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="flex gap-2 border-t border-slate-200 px-3 py-3 dark:border-slate-700">
            <input
              type="text"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
              placeholder="Ask about loans..."
              className="min-w-0 flex-1 rounded-lg border border-slate-300 bg-slate-50 px-3 py-2
                text-sm text-slate-900 outline-none transition placeholder:text-slate-400
                focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 disabled:opacity-50
                dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
            />
            <button
              type="button"
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white
                transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Send
            </button>
          </div>
        </section>
      )}

      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        aria-label={open ? 'Close financial assistant' : 'Open financial assistant'}
        className="grid h-14 w-14 place-items-center rounded-full bg-indigo-600 text-sm
          font-bold text-white shadow-xl transition hover:bg-indigo-700 focus:outline-none
          focus:ring-4 focus:ring-indigo-300"
      >
        {open ? 'Close' : 'Chat'}
      </button>
    </div>
  )
}
