/**
 * LoanForm.jsx
 *
 * Controlled form component for the credit-risk dataset feature set.
 */

import { useMemo, useState } from 'react'

const INITIAL_STATE = {
  person_age: '',
  person_income: '',
  person_home_ownership: 'RENT',
  person_emp_length: '',
  loan_intent: 'PERSONAL',
  loan_grade: 'C',
  loan_amnt: '',
  loan_int_rate: '',
  cb_person_default_on_file: 'N',
  cb_person_cred_hist_length: '',
}

const FIELDS = [
  {
    key: 'person_age',
    label: 'Age',
    type: 'number',
    placeholder: '35',
    min: 18,
    step: 1,
    helpText: 'Applicant age in years',
  },
  {
    key: 'person_income',
    label: 'Annual Income (USD)',
    type: 'number',
    placeholder: '59000',
    min: 1,
    step: 1000,
    helpText: 'Gross annual income',
  },
  {
    key: 'person_home_ownership',
    label: 'Home Ownership',
    type: 'select',
    options: ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
    helpText: 'Current housing status',
  },
  {
    key: 'person_emp_length',
    label: 'Employment Length (years)',
    type: 'number',
    placeholder: '3',
    min: 0,
    step: 0.5,
    helpText: 'Years in current employment',
  },
  {
    key: 'loan_intent',
    label: 'Loan Purpose',
    type: 'select',
    options: [
      'PERSONAL',
      'EDUCATION',
      'MEDICAL',
      'VENTURE',
      'HOMEIMPROVEMENT',
      'DEBTCONSOLIDATION',
    ],
    helpText: 'Primary reason for borrowing',
  },
  {
    key: 'loan_grade',
    label: 'Loan Grade',
    type: 'select',
    options: ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    helpText: 'Credit grade assigned to the loan',
  },
  {
    key: 'loan_amnt',
    label: 'Loan Amount (USD)',
    type: 'number',
    placeholder: '35000',
    min: 1,
    step: 500,
    helpText: 'Total amount requested',
  },
  {
    key: 'loan_int_rate',
    label: 'Interest Rate (%)',
    type: 'number',
    placeholder: '15.23',
    min: 0,
    step: 0.01,
    helpText: 'Quoted annual interest rate',
  },
  {
    key: 'cb_person_default_on_file',
    label: 'Previously Defaulted',
    type: 'select',
    options: ['N', 'Y'],
    helpText: 'Whether a prior default is on file',
  },
  {
    key: 'cb_person_cred_hist_length',
    label: 'Credit History Length (years)',
    type: 'number',
    placeholder: '3',
    min: 0,
    step: 1,
    helpText: 'Length of credit history',
  },
]

const NUMERIC_FIELDS = new Set([
  'person_age',
  'person_income',
  'person_emp_length',
  'loan_amnt',
  'loan_int_rate',
  'cb_person_cred_hist_length',
])

export default function LoanForm({ onSubmit, loading }) {
  const [values, setValues] = useState(INITIAL_STATE)
  const [errors, setErrors] = useState({})

  const loanPercentIncome = useMemo(() => {
    const income = parseFloat(values.person_income)
    const amount = parseFloat(values.loan_amnt)
    if (!Number.isFinite(income) || income <= 0 || !Number.isFinite(amount)) {
      return ''
    }
    return Math.min(amount / income, 1).toFixed(2)
  }, [values.person_income, values.loan_amnt])

  const validate = () => {
    const nextErrors = {}
    FIELDS.forEach(({ key, min, max }) => {
      const value = values[key]
      if (value === '' || value === null || typeof value === 'undefined') {
        nextErrors[key] = 'This field is required.'
        return
      }

      if (NUMERIC_FIELDS.has(key)) {
        const numericValue = parseFloat(value)
        if (!Number.isFinite(numericValue)) {
          nextErrors[key] = 'Enter a valid number.'
        } else if (min !== undefined && numericValue < min) {
          nextErrors[key] = `Minimum value is ${min}.`
        } else if (max !== undefined && numericValue > max) {
          nextErrors[key] = `Maximum value is ${max}.`
        }
      }
    })
    return nextErrors
  }

  const handleChange = (key) => (event) => {
    setValues((prev) => ({ ...prev, [key]: event.target.value }))
    if (errors[key]) {
      setErrors((prev) => ({ ...prev, [key]: undefined }))
    }
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    const validationErrors = validate()
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors)
      return
    }

    const personIncome = parseFloat(values.person_income)
    const loanAmount = parseFloat(values.loan_amnt)
    const payload = {
      person_age: parseInt(values.person_age, 10),
      person_income: personIncome,
      person_home_ownership: values.person_home_ownership,
      person_emp_length: parseFloat(values.person_emp_length),
      loan_intent: values.loan_intent,
      loan_grade: values.loan_grade,
      loan_amnt: loanAmount,
      loan_int_rate: parseFloat(values.loan_int_rate),
      loan_percent_income: Math.min(loanAmount / personIncome, 1),
      cb_person_default_on_file: values.cb_person_default_on_file,
      cb_person_cred_hist_length: parseFloat(values.cb_person_cred_hist_length),
    }

    onSubmit(payload)
  }

  return (
    <form onSubmit={handleSubmit} noValidate className="space-y-5">
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
        {FIELDS.map(({ key, label, placeholder, min, max, step, type, options, helpText }) => (
          <div key={key} className="flex flex-col gap-1">
            <label
              htmlFor={key}
              className="text-sm font-semibold text-slate-700 dark:text-slate-300"
            >
              {label}
            </label>

            {type === 'select' ? (
              <select
                id={key}
                value={values[key]}
                onChange={handleChange(key)}
                disabled={loading}
                className={`rounded-lg border px-4 py-2.5 text-sm bg-white dark:bg-slate-800
                  text-slate-900 dark:text-slate-100 outline-none transition
                  focus:ring-2 focus:ring-indigo-500 disabled:opacity-50
                  disabled:cursor-not-allowed
                  ${errors[key] ? 'border-red-500 focus:ring-red-400' : 'border-slate-300 dark:border-slate-600'}`}
              >
                {options.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            ) : (
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
                  transition focus:ring-2 focus:ring-indigo-500 disabled:opacity-50
                  disabled:cursor-not-allowed
                  ${errors[key] ? 'border-red-500 focus:ring-red-400' : 'border-slate-300 dark:border-slate-600'}`}
              />
            )}

            {errors[key] ? (
              <p className="text-xs text-red-500">{errors[key]}</p>
            ) : (
              <p className="text-xs text-slate-400 dark:text-slate-500">{helpText}</p>
            )}
          </div>
        ))}

        <div className="flex flex-col gap-1">
          <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">
            Loan Percent of Income
          </label>
          <div className="rounded-lg border border-slate-300 dark:border-slate-600 px-4 py-2.5 text-sm bg-slate-50 dark:bg-slate-900 text-slate-700 dark:text-slate-200">
            {loanPercentIncome === '' ? 'Calculated from income and amount' : loanPercentIncome}
          </div>
          <p className="text-xs text-slate-400 dark:text-slate-500">
            Automatically included in the prediction
          </p>
        </div>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full rounded-xl bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800
          text-white font-semibold py-3 px-6 transition disabled:opacity-60
          disabled:cursor-not-allowed focus:outline-none focus:ring-4 focus:ring-indigo-400"
      >
        {loading ? 'Analyzing...' : 'Check Eligibility'}
      </button>
    </form>
  )
}
