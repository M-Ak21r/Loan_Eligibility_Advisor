/**
 * LoanForm.jsx
 *
 * Controlled form component for capturing loan application inputs.
 * Calls onSubmit with the validated payload when the user submits.
 */

import { useState } from 'react'

const INITIAL_STATE = {
  person_age: '',
  person_income: '',
  person_home_ownership: 'RENT',
  person_emp_length: '',
  loan_intent: 'PERSONAL',
  loan_grade: 'C',
  loan_amnt: '',
  loan_int_rate: '',
  loan_percent_income: '',
  cb_person_default_on_file: 'N',
  cb_person_cred_hist_length: '',
}

const FIELDS = [
  { key: 'person_age', label: 'Age', type: 'number', placeholder: '35', min: 18, step: 1 },
  { key: 'person_income', label: 'Annual Income (USD)', type: 'number', placeholder: '59000', min: 0, step: 100 },
  {
    key: 'person_home_ownership',
    label: 'Home Ownership',
    type: 'select',
    options: ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
  },
  { key: 'person_emp_length', label: 'Employment Length (years)', type: 'number', placeholder: '3.0', min: 0, step: 0.5 },
  {
    key: 'loan_intent',
    label: 'Loan Purpose',
    type: 'select',
    options: ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOME_IMPROVEMENT', 'DEBT_CONSOLIDATION', 'OTHER'],
  },
  {
    key: 'loan_grade',
    label: 'Loan Grade',
    type: 'select',
    options: ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
  },
  { key: 'loan_amnt', label: 'Loan Amount (USD)', type: 'number', placeholder: '35000', min: 0, step: 100 },
  { key: 'loan_int_rate', label: 'Interest Rate (%)', type: 'number', placeholder: '15.23', min: 0, step: 0.01 },
  { key: 'loan_percent_income', label: 'Loan % of Income', type: 'number', placeholder: '0.59', min: 0, step: 0.01 },
  {
    key: 'cb_person_default_on_file',
    label: 'Previously Defaulted',
    type: 'select',
    options: ['Y', 'N'],
  },
  { key: 'cb_person_cred_hist_length', label: 'Credit History Length (years)', type: 'number', placeholder: '3', min: 0, step: 1 },
]

export default function LoanForm({ onSubmit, loading }) {
  const [values, setValues] = useState(INITIAL_STATE)
  const [errors, setErrors] = useState({})

  const validate = () => {
    const newErrors = {}
    FIELDS.forEach(({ key }) => {
      const val = values[key]
      if (val === '' || val === null || typeof val === 'undefined') {
        newErrors[key] = 'This field is required.'
      }
    })
    return newErrors
  }

  const handleChange = (key) => (e) => {
    const v = e.target.value
    setValues((prev) => ({ ...prev, [key]: v }))
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
      person_age: parseInt(values.person_age, 10),
      person_income: parseFloat(values.person_income),
      person_home_ownership: values.person_home_ownership,
      person_emp_length: parseFloat(values.person_emp_length),
      loan_intent: values.loan_intent,
      loan_grade: values.loan_grade,
      loan_amnt: parseFloat(values.loan_amnt),
      loan_int_rate: parseFloat(values.loan_int_rate),
      loan_percent_income: parseFloat(values.loan_percent_income),
      cb_person_default_on_file: values.cb_person_default_on_file,
      cb_person_cred_hist_length: parseFloat(values.cb_person_cred_hist_length),
    }

    onSubmit(payload)
  }

  return (
    <form onSubmit={handleSubmit} noValidate className="space-y-5">
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
        {FIELDS.map(({ key, label, placeholder, min, step, type, options }) => (
          <div key={key} className="flex flex-col gap-1">
            <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">{label}</label>

            {type === 'select' ? (
              <select value={values[key]} onChange={handleChange(key)} disabled={loading}
                className="rounded-lg border px-4 py-2.5 text-sm bg-white dark:bg-slate-800">
                {options.map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
            ) : (
              <input
                type="number"
                placeholder={placeholder}
                min={min}
                step={step}
                value={values[key]}
                onChange={handleChange(key)}
                disabled={loading}
                className="rounded-lg border px-4 py-2.5 text-sm bg-white dark:bg-slate-800"
              />
            )}

            {errors[key] ? <p className="text-xs text-red-500">{errors[key]}</p> : null}
          </div>
        ))}
      </div>

      <button type="submit" disabled={loading} className="w-full rounded-xl bg-indigo-600 text-white py-3">
        {loading ? 'Analyzing…' : 'Check Eligibility'}
      </button>
    </form>
  )
}
