/**
 * ResultCard.jsx
 *
 * Displays the API prediction result: decision badge, probability bar,
 * and a list of identified risk factors.
 */

const DECISION_STYLES = {
  Approved: {
    bg: 'bg-emerald-50 dark:bg-emerald-900/30',
    border: 'border-emerald-400 dark:border-emerald-600',
    badge: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-800/50 dark:text-emerald-200',
    icon: '✅',
    barColor: 'bg-emerald-500',
  },
  Conditional: {
    bg: 'bg-amber-50 dark:bg-amber-900/30',
    border: 'border-amber-400 dark:border-amber-600',
    badge: 'bg-amber-100 text-amber-800 dark:bg-amber-800/50 dark:text-amber-200',
    icon: '⚠️',
    barColor: 'bg-amber-500',
  },
  Rejected: {
    bg: 'bg-red-50 dark:bg-red-900/30',
    border: 'border-red-400 dark:border-red-600',
    badge: 'bg-red-100 text-red-800 dark:bg-red-800/50 dark:text-red-200',
    icon: '❌',
    barColor: 'bg-red-500',
  },
}

export default function ResultCard({ result }) {
  if (!result) return null

  const { decision, probability_approved, risk_factors } = result
  const style = DECISION_STYLES[decision] ?? DECISION_STYLES['Conditional']
  const pct = (probability_approved * 100).toFixed(1)

  return (
    <div
      className={`rounded-2xl border-2 p-6 space-y-5 transition-all ${style.bg} ${style.border}`}
      role="region"
      aria-label="Loan eligibility result"
    >
      {/* Decision badge */}
      <div className="flex items-center gap-3">
        <span className="text-3xl" role="img" aria-label={decision}>
          {style.icon}
        </span>
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 dark:text-slate-400">
            Decision
          </p>
          <span
            className={`inline-block mt-0.5 rounded-full px-3 py-0.5 text-sm font-bold ${style.badge}`}
          >
            {decision}
          </span>
        </div>
      </div>

      {/* Approval probability */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-sm text-slate-600 dark:text-slate-300">
          <span className="font-medium">Approval Probability</span>
          <span className="font-bold">{pct}%</span>
        </div>
        <div className="h-3 w-full rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
          <div
            className={`h-3 rounded-full transition-all duration-700 ${style.barColor}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Risk factors */}
      <div>
        <p className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
          Risk Factors Identified
        </p>
        {risk_factors.length === 0 ? (
          <p className="text-sm text-slate-500 dark:text-slate-400 italic">
            No major risk factors detected.
          </p>
        ) : (
          <ul className="space-y-1.5">
            {risk_factors.map((factor) => (
              <li
                key={factor}
                className="flex items-start gap-2 text-sm text-slate-700 dark:text-slate-200"
              >
                <span className="mt-0.5 text-red-500 font-bold shrink-0">•</span>
                {factor}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
