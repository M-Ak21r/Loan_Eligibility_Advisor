/**
 * LoanForm.jsx
 *
 * Controlled form component for capturing loan application inputs.
 * Calls onSubmit with the validated payload when the user submits.
 */

import { useState } from 'react'

const INITIAL_STATE = {
  annual_income: '',
  credit_score: '',
  debt_to_income_ratio: '',
  employment_length_years: '',
  loan_amount_requested: '',
}

const FIELDS = [
  {
    key: 'annual_income',
    label: 'Annual Income (USD)',
    placeholder: '75000',
    min: 1,
    step: 1000,
    helpText: 'Gross annual income before taxes',
  },
  {
    key: 'credit_score',
    label: 'Credit Score',
    placeholder: '720',
    min: 300,
    max: 850,
    step: 1,
    helpText: 'FICO score between 300 and 850',
  },
  {
    key: 'debt_to_income_ratio',
    label: 'Debt-to-Income Ratio',
    placeholder: '0.25',
    min: 0,
    max: 1,
    step: 0.01,
    helpText: 'Total monthly debt divided by gross monthly income (0.00 – 1.00)',
  },
  {
    key: 'employment_length_years',
    label: 'Employment Length (years)',
    placeholder: '5',
    min: 0,
    step: 0.5,
    helpText: 'Years in current or most-recent job',
  },
  {
    key: 'loan_amount_requested',
    label: 'Loan Amount Requested (USD)',
    placeholder: '15000',
    min: 1,
    step: 500,
    helpText: 'Total amount you wish to borrow',
  },
]

export default function LoanForm({ onSubmit, loading }) {
  const [values, setValues] = useState(INITIAL_STATE)
  const [errors, setErrors] = useState({})

  const validate = () => {
    const newErrors = {}
    FIELDS.forEach(({ key, min, max }) => {
      const val = parseFloat(values[key])
      if (values[key] === '' || isNaN(val)) {
        newErrors[key] = 'This field is required.'
      } else if (min !== undefined && val < min) {
        newErrors[key] = `Minimum value is ${min}.`
      } else if (max !== undefined && val > max) {
        newErrors[key] = `Maximum value is ${max}.`
      }
    })
    return newErrors
  }

  const handleChange = (key) => (e) => {
    setValues((prev) => ({ ...prev, [key]: e.target.value }))
    if (errors[key]) setErrors((prev) => ({ ...prev, [key]: undefined }))
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    const validationErrors = validate()
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors)
      return
    }
    const payload = {
      annual_income: parseFloat(values.annual_income),
      credit_score: parseFloat(values.credit_score),
      debt_to_income_ratio: parseFloat(values.debt_to_income_ratio),
      employment_length_years: parseFloat(values.employment_length_years),
      loan_amount_requested: parseFloat(values.loan_amount_requested),
    }
    onSubmit(payload)
  }

  return (
    <form onSubmit={handleSubmit} noValidate className="space-y-5">
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
        {FIELDS.map(({ key, label, placeholder, min, max, step, helpText }) => (
          <div key={key} className="flex flex-col gap-1">
            <label
              htmlFor={key}
              className="text-sm font-semibold text-slate-700 dark:text-slate-300"
            >
              {label}
            </label>
            <input
              id={key}
              type="number"
              placeholder={placeholder}
              min={min}
              max={max}
              step={step}
              value={values[key]}
              onChange={handleChange(key)}
              disabled={loading}
              className={`rounded-lg border px-4 py-2.5 text-sm bg-white dark:bg-slate-800
                text-slate-900 dark:text-slate-100 placeholder-slate-400 outline-none
                transition focus:ring-2 focus:ring-indigo-500
                disabled:opacity-50 disabled:cursor-not-allowed
                ${
                  errors[key]
                    ? 'border-red-500 focus:ring-red-400'
                    : 'border-slate-300 dark:border-slate-600'
                }`}
            />
            {errors[key] ? (
              <p className="text-xs text-red-500">{errors[key]}</p>
            ) : (
              <p className="text-xs text-slate-400 dark:text-slate-500">{helpText}</p>
            )}
          </div>
        ))}
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full rounded-xl bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800
          text-white font-semibold py-3 px-6 transition
          disabled:opacity-60 disabled:cursor-not-allowed
          focus:outline-none focus:ring-4 focus:ring-indigo-400"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg
              className="animate-spin h-4 w-4 text-white"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8v8H4z"
              />
            </svg>
            Analyzing…
          </span>
        ) : (
          'Check Eligibility'
        )}
      </button>
    </form>
  )
}
