'use client'

import { useEffect, useRef, useState } from 'react'
import AppShell from '../../components/AppShell'
import { createAccount, fetchAccounts, isSessionExpired, type AccountInfo } from '../../lib/api'
import { useAuthGuard } from '../../lib/useAuthGuard'

function formatTimestamp(value: string | null) {
  if (!value) return '—'
  return new Date(value).toLocaleString()
}

export default function AccountsPage() {
  const { user, isChecking } = useAuthGuard({ requireAdmin: true })
  const [accounts, setAccounts] = useState<AccountInfo[]>([])
  const [errorMessage, setErrorMessage] = useState('')

  const [newAccountEmail, setNewAccountEmail] = useState('')
  const [newAccountName, setNewAccountName] = useState('')
  const [newAccountPassword, setNewAccountPassword] = useState('')
  const [newAccountIsAdmin, setNewAccountIsAdmin] = useState(false)
  const [isCreatingAccount, setIsCreatingAccount] = useState(false)
  const [formHint, setFormHint] = useState('')
  const hintTimerRef = useRef<number | null>(null)

  function showFormHint(message: string) {
    setFormHint(message)
    if (hintTimerRef.current) window.clearTimeout(hintTimerRef.current)
    hintTimerRef.current = window.setTimeout(() => setFormHint(''), 4000)
  }

  function handleRequestError(error: unknown, fallbackMessage: string) {
    if (isSessionExpired(error)) return
    setErrorMessage(error instanceof Error ? error.message : fallbackMessage)
  }

  async function loadAccounts() {
    try {
      setAccounts((await fetchAccounts()).accounts)
    } catch (error) {
      handleRequestError(error, 'Failed to load accounts.')
    }
  }

  async function handleCreateAccount() {
    const missing = []
    if (!newAccountEmail.trim()) missing.push('an email')
    if (!newAccountName.trim()) missing.push('a display name')
    if (newAccountPassword.length < 8) missing.push('a password of at least 8 characters')
    if (missing.length) {
      showFormHint(`Still needed: ${missing.join(', ')}.`)
      return
    }
    setIsCreatingAccount(true)
    setErrorMessage('')
    try {
      await createAccount({
        email: newAccountEmail.trim(),
        display_name: newAccountName.trim(),
        password: newAccountPassword,
        is_admin: newAccountIsAdmin
      })
      setNewAccountEmail('')
      setNewAccountName('')
      setNewAccountPassword('')
      setNewAccountIsAdmin(false)
      await loadAccounts()
    } catch (error) {
      handleRequestError(error, 'Failed to create the account.')
    } finally {
      setIsCreatingAccount(false)
    }
  }

  useEffect(() => {
    if (isChecking || !user) return
    void loadAccounts()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isChecking, user])

  useEffect(() => () => {
    if (hintTimerRef.current) window.clearTimeout(hintTimerRef.current)
  }, [])

  if (isChecking) {
    return (
      <div className="auth-shell">
        <div className="auth-card auth-card-compact">
          <p className="eyebrow">Authentication</p>
          <h1>Checking session</h1>
          <p className="auth-copy">Validating your login before loading accounts.</p>
        </div>
      </div>
    )
  }

  if (!user) return null

  return (
    <AppShell user={user}>
      <header className="hero hero-insights">
        <div className="hero-main">
          <div>
            <p className="eyebrow">Accounts</p>
            <h2>Dashboard logins</h2>
            <p>
              The people who sign in to this dashboard — care partners and the research team.
              Linking an account to a profile happens on the Profiles page.
            </p>
          </div>

          <div className="hero-filters hero-filters-insights">
            <div className="toolbar-meta">
              <strong>{accounts.length}</strong> accounts
            </div>
          </div>
        </div>
      </header>

      {errorMessage ? <section className="error-banner">{errorMessage}</section> : null}

      <div className="patients-layout">
        <section className="timeline-panel insights-panel">
          <div className="panel-header">
            <h3>Accounts</h3>
          </div>

          {accounts.length ? (
            <table className="data-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Email</th>
                  <th>Name</th>
                  <th>Admin</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {accounts.map((account) => (
                  <tr key={account.id}>
                    <td>{account.id}</td>
                    <td>{account.email}</td>
                    <td>{account.display_name}</td>
                    <td>{account.is_admin ? <span className="pill">admin</span> : '—'}</td>
                    <td>{formatTimestamp(account.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="empty-state">No accounts yet.</div>
          )}

          <div className="panel-divider" />

          <div className="panel-header">
            <h3>New account</h3>
          </div>
          <div className="settings-grid">
            <label className="field">
              <span>Email</span>
              <input
                autoComplete="off"
                onChange={(event) => setNewAccountEmail(event.target.value)}
                placeholder="carepartner@example.org"
                type="email"
                value={newAccountEmail}
              />
            </label>
            <label className="field">
              <span>Display name</span>
              <input
                autoComplete="off"
                onChange={(event) => setNewAccountName(event.target.value)}
                type="text"
                value={newAccountName}
              />
            </label>
            <label className="field">
              <span>Password (min 8 characters)</span>
              <input
                autoComplete="new-password"
                onChange={(event) => setNewAccountPassword(event.target.value)}
                type="password"
                value={newAccountPassword}
              />
            </label>
          </div>
          <label className="auth-checkbox">
            <input
              checked={newAccountIsAdmin}
              onChange={(event) => setNewAccountIsAdmin(event.target.checked)}
              type="checkbox"
            />
            <span>Administrator (sees all profiles, can provision)</span>
          </label>
          <div className="inline-actions">
            <button
              className="send-button"
              disabled={isCreatingAccount}
              onClick={() => void handleCreateAccount()}
              type="button"
            >
              {isCreatingAccount ? 'Creating…' : 'Create account'}
            </button>
            {formHint ? <p className="form-hint">{formHint}</p> : null}
          </div>
        </section>
      </div>
    </AppShell>
  )
}
