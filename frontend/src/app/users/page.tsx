'use client'

import { useEffect, useState } from 'react'
import { useForm, useStore } from '@tanstack/react-form'
import AppShell from '../../components/AppShell'
import {
  createAccount,
  createProfile,
  fetchAccounts,
  fetchProfileLinks,
  fetchProfiles,
  fetchTokens,
  issueDeviceToken,
  isSessionExpired,
  linkAccountProfile,
  patchProfile,
  revokeToken,
  unlinkAccountProfile,
  type AccountInfo,
  type DeviceTokenIssue,
  type JsonObject,
  type LinkInfo,
  type Profile,
  type TokenInfo
} from '../../lib/api'
import { useAuthGuard } from '../../lib/useAuthGuard'

const SELECTED_PROFILE_STORAGE_KEY = 'robin-selected-profile-id'

// The full IANA list from the browser itself, with a static fallback for older ones.
// US zones first — this is where profiles actually live.
const US_TIMEZONES = [
  'America/New_York',
  'America/Chicago',
  'America/Denver',
  'America/Phoenix',
  'America/Los_Angeles',
  'America/Anchorage',
  'Pacific/Honolulu'
]

function allTimezones(): string[] {
  try {
    const supported = Intl.supportedValuesOf('timeZone')
    return [...US_TIMEZONES, ...supported.filter((tz) => !US_TIMEZONES.includes(tz))]
  } catch {
    return US_TIMEZONES
  }
}

// Kokoro-82M voices (fetched on demand by the server). Prefix decodes as
// a=American/b=British + f=female/m=male.
const KOKORO_VOICES = [
  { id: 'af_heart', label: 'Heart — American female (default)' },
  { id: 'af_bella', label: 'Bella — American female' },
  { id: 'af_nicole', label: 'Nicole — American female' },
  { id: 'af_sarah', label: 'Sarah — American female' },
  { id: 'af_sky', label: 'Sky — American female' },
  { id: 'am_adam', label: 'Adam — American male' },
  { id: 'am_michael', label: 'Michael — American male' },
  { id: 'am_eric', label: 'Eric — American male' },
  { id: 'bf_emma', label: 'Emma — British female' },
  { id: 'bf_isabella', label: 'Isabella — British female' },
  { id: 'bm_george', label: 'George — British male' },
  { id: 'bm_lewis', label: 'Lewis — British male' }
]

const EMPTY_PROFILE_FORM = {
  displayName: '',
  timezone: '',
  voice: '',
  speechRate: '0.85',
  contextJson: ''
}

function buildProfileForm(profile: Profile) {
  return {
    displayName: profile.display_name,
    timezone: profile.timezone,
    voice: profile.voice,
    speechRate: String(profile.speech_rate),
    contextJson: Object.keys(profile.context).length
      ? JSON.stringify(profile.context, null, 2)
      : ''
  }
}

function canUseStorage() {
  return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined'
}

function getStoredSelectedProfileId() {
  if (!canUseStorage()) return null
  const rawValue = window.localStorage.getItem(SELECTED_PROFILE_STORAGE_KEY)
  if (!rawValue) return null
  const parsedValue = Number(rawValue)
  return Number.isFinite(parsedValue) ? parsedValue : null
}

function formatTimestamp(value: string | null) {
  if (!value) return '—'
  return new Date(value).toLocaleString()
}

export default function UsersPage() {
  const { user, isChecking } = useAuthGuard({ requireAdmin: true })
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [accounts, setAccounts] = useState<AccountInfo[]>([])
  const [links, setLinks] = useState<LinkInfo[]>([])
  const [tokens, setTokens] = useState<TokenInfo[]>([])
  const [selectedProfileId, setSelectedProfileId] = useState<number | null>(null)
  const [isCreatingProfile, setIsCreatingProfile] = useState(false)
  const [isEditingProfile, setIsEditingProfile] = useState(false)
  const [issuedToken, setIssuedToken] = useState<DeviceTokenIssue | null>(null)
  const [errorMessage, setErrorMessage] = useState('')

  // Account create form (plain controlled inputs; only the profile form needs react-form).
  const [newAccountEmail, setNewAccountEmail] = useState('')
  const [newAccountName, setNewAccountName] = useState('')
  const [newAccountPassword, setNewAccountPassword] = useState('')
  const [newAccountIsAdmin, setNewAccountIsAdmin] = useState(false)
  const [isCreatingAccount, setIsCreatingAccount] = useState(false)

  // Link + token forms.
  const [linkAccountId, setLinkAccountId] = useState('')
  const [linkRole, setLinkRole] = useState<'owner' | 'viewer'>('owner')
  const [tokenLabel, setTokenLabel] = useState('')

  const selectedProfile = profiles.find((profile) => profile.id === selectedProfileId) || null

  const form = useForm({
    defaultValues: EMPTY_PROFILE_FORM,

    onSubmit: async ({ value, formApi }) => {
      let context: JsonObject = {}
      if (value.contextJson.trim()) {
        try {
          const parsed = JSON.parse(value.contextJson)
          if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
            throw new Error('not an object')
          }
          context = parsed as JsonObject
        } catch {
          setErrorMessage('Context must be a JSON object, e.g. {"likes": "gardening"}.')
          return
        }
      }
      const speechRate = value.speechRate.trim() ? Number(value.speechRate) : null
      if (speechRate !== null && (!Number.isFinite(speechRate) || speechRate <= 0)) {
        setErrorMessage('Speech rate must be a positive number.')
        return
      }

      setErrorMessage('')
      try {
        if (isCreatingProfile) {
          const created = await createProfile({
            display_name: value.displayName.trim(),
            ...(value.timezone.trim() ? { timezone: value.timezone.trim() } : {}),
            ...(value.voice.trim() ? { voice: value.voice.trim() } : {}),
            ...(speechRate !== null ? { speech_rate: speechRate } : {}),
            ...(Object.keys(context).length ? { context } : {})
          })
          setIsCreatingProfile(false)
          await loadProfiles()
          setSelectedProfileId(created.id)
        } else if (selectedProfileId) {
          await patchProfile(selectedProfileId, {
            display_name: value.displayName.trim(),
            timezone: value.timezone.trim(),
            voice: value.voice.trim(),
            ...(speechRate !== null ? { speech_rate: speechRate } : {}),
            context
          })
          formApi.reset(value)
          setIsEditingProfile(false)
          await loadProfiles()
        }
      } catch (error) {
        handleRequestError(error, 'Failed to save the profile.')
      }
    }
  })

  // `form.state` is not reactive on its own, so subscribe to the pieces this page renders.
  const isSubmitting = useStore(form.store, (state) => state.isSubmitting)
  const isPristine = useStore(form.store, (state) => state.isPristine)

  function handleRequestError(error: unknown, fallbackMessage: string) {
    if (isSessionExpired(error)) return // useAuthGuard's next fetchMe redirects to /login
    setErrorMessage(error instanceof Error ? error.message : fallbackMessage)
  }

  async function loadProfiles() {
    try {
      const response = await fetchProfiles()
      setProfiles(response.profiles)
      setSelectedProfileId((current) => {
        const candidateId = current ?? getStoredSelectedProfileId()
        if (candidateId && response.profiles.some((profile) => profile.id === candidateId)) {
          return candidateId
        }
        return response.profiles[0]?.id ?? null
      })
    } catch (error) {
      handleRequestError(error, 'Failed to load profiles.')
    }
  }

  async function loadAccounts() {
    try {
      setAccounts((await fetchAccounts()).accounts)
    } catch (error) {
      handleRequestError(error, 'Failed to load accounts.')
    }
  }

  async function loadLinksAndTokens(profileId: number) {
    try {
      const [linksResponse, tokensResponse] = await Promise.all([
        fetchProfileLinks(profileId),
        fetchTokens({ profile_id: profileId, include_revoked: true })
      ])
      setLinks(linksResponse.links)
      setTokens(tokensResponse.tokens)
    } catch (error) {
      handleRequestError(error, 'Failed to load links and tokens.')
    }
  }

  async function handleCreateAccount() {
    if (!newAccountEmail.trim() || !newAccountName.trim() || !newAccountPassword) return
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

  async function handleLink() {
    if (!selectedProfileId || !linkAccountId) return
    setErrorMessage('')
    try {
      await linkAccountProfile(selectedProfileId, {
        account_id: Number(linkAccountId),
        role: linkRole
      })
      setLinkAccountId('')
      await loadLinksAndTokens(selectedProfileId)
    } catch (error) {
      handleRequestError(error, 'Failed to link the account.')
    }
  }

  async function handleUnlink(accountId: number) {
    if (!selectedProfileId) return
    setErrorMessage('')
    try {
      await unlinkAccountProfile(selectedProfileId, accountId)
      await loadLinksAndTokens(selectedProfileId)
    } catch (error) {
      handleRequestError(error, 'Failed to unlink the account.')
    }
  }

  async function handleIssueToken() {
    if (!selectedProfileId || !tokenLabel.trim()) return
    setErrorMessage('')
    try {
      const issued = await issueDeviceToken(selectedProfileId, tokenLabel.trim())
      setIssuedToken(issued)
      setTokenLabel('')
      await loadLinksAndTokens(selectedProfileId)
    } catch (error) {
      handleRequestError(error, 'Failed to issue a device token.')
    }
  }

  async function handleRevokeToken(tokenId: number) {
    if (!selectedProfileId) return
    setErrorMessage('')
    try {
      await revokeToken(tokenId)
      if (issuedToken?.token_id === tokenId) setIssuedToken(null)
      await loadLinksAndTokens(selectedProfileId)
    } catch (error) {
      handleRequestError(error, 'Failed to revoke the token.')
    }
  }

  function handleStartCreateProfile() {
    setIsCreatingProfile(true)
    setIsEditingProfile(false)
    setErrorMessage('')
    form.reset(EMPTY_PROFILE_FORM)
  }

  function handleStartEditProfile() {
    if (!selectedProfile) return
    setIsEditingProfile(true)
    setErrorMessage('')
  }

  function handleCancelProfileForm() {
    setIsCreatingProfile(false)
    setIsEditingProfile(false)
    setErrorMessage('')
    form.reset(selectedProfile ? buildProfileForm(selectedProfile) : EMPTY_PROFILE_FORM)
  }

  useEffect(() => {
    if (isChecking || !user) return
    void loadProfiles()
    void loadAccounts()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isChecking, user])

  useEffect(() => {
    setIssuedToken(null) // a raw token is only ever shown for the profile it was issued to
    setIsCreatingProfile(false)
    setIsEditingProfile(false)
    if (!selectedProfileId) {
      setLinks([])
      setTokens([])
      form.reset(EMPTY_PROFILE_FORM)
      return
    }
    void loadLinksAndTokens(selectedProfileId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProfileId])

  useEffect(() => {
    if (isCreatingProfile || isEditingProfile) return
    // Never clobber edits that are still in flight.
    if (!form.state.isPristine) return
    form.reset(selectedProfile ? buildProfileForm(selectedProfile) : EMPTY_PROFILE_FORM)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProfile, isCreatingProfile, isEditingProfile])

  useEffect(() => {
    if (!canUseStorage()) return
    if (selectedProfileId == null) {
      window.localStorage.removeItem(SELECTED_PROFILE_STORAGE_KEY)
      return
    }
    window.localStorage.setItem(SELECTED_PROFILE_STORAGE_KEY, String(selectedProfileId))
  }, [selectedProfileId])

  if (isChecking) {
    return (
      <div className="auth-shell">
        <div className="auth-card auth-card-compact">
          <p className="eyebrow">Authentication</p>
          <h1>Checking session</h1>
          <p className="auth-copy">Validating your login before loading users.</p>
        </div>
      </div>
    )
  }

  if (!user) return null

  const formDisabled = (!isCreatingProfile && !isEditingProfile) || isSubmitting
  const accountById = new Map(accounts.map((account) => [account.id, account]))

  return (
    <AppShell user={user}>
      <header className="hero hero-insights">
        <div className="hero-main">
          <div>
            <p className="eyebrow">Users</p>
            <h2>{isCreatingProfile ? 'New profile' : selectedProfile?.display_name || 'Profiles'}</h2>
            <p>Provision the people Robin talks to and the accounts that watch over them.</p>
          </div>

          <div className="hero-filters hero-filters-insights">
            <label className="field field-patient">
              <span>Profile</span>
              <select
                value={selectedProfileId ?? ''}
                onChange={(event) =>
                  setSelectedProfileId(event.target.value ? Number(event.target.value) : null)
                }
              >
                <option value="">Select a profile</option>
                {profiles.map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.id} · {profile.display_name}
                  </option>
                ))}
              </select>
            </label>

            <button className="secondary-button" onClick={handleStartCreateProfile} type="button">
              New profile
            </button>

            <div className="toolbar-meta">
              <strong>{profiles.length}</strong> profiles · <strong>{accounts.length}</strong>{' '}
              accounts
            </div>
          </div>
        </div>
      </header>

      {errorMessage ? <section className="error-banner">{errorMessage}</section> : null}

      <div className="patients-layout">
        <section className="timeline-panel insights-panel">
          <div className="panel-header">
            <h3>{isCreatingProfile ? 'New profile' : 'Profile details'}</h3>
            <div className="inline-actions">
              {isCreatingProfile || isEditingProfile ? (
                <>
                  <button
                    className="send-button"
                    disabled={isSubmitting || (isEditingProfile && isPristine)}
                    onClick={() => void form.handleSubmit()}
                    type="button"
                  >
                    {isSubmitting ? 'Saving…' : isCreatingProfile ? 'Create' : 'Save'}
                  </button>
                  <button
                    className="secondary-button"
                    disabled={isSubmitting}
                    onClick={handleCancelProfileForm}
                    type="button"
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <button
                  className="secondary-button"
                  disabled={!selectedProfile}
                  onClick={handleStartEditProfile}
                  type="button"
                >
                  Edit
                </button>
              )}
            </div>
          </div>

          {!selectedProfile && !isCreatingProfile ? (
            <div className="empty-state">Select a profile or create a new one.</div>
          ) : (
            <>
              <div className="settings-grid">
                <label className="field">
                  <span>Display name</span>
                  <form.Field name="displayName">
                    {(field) => (
                      <input
                        disabled={formDisabled}
                        onChange={(event) => field.handleChange(event.target.value)}
                        placeholder="Margaret"
                        type="text"
                        value={field.state.value}
                      />
                    )}
                  </form.Field>
                </label>
                <label className="field">
                  <span>Timezone</span>
                  <form.Field name="timezone">
                    {(field) => (
                      <select
                        disabled={formDisabled}
                        onChange={(event) => field.handleChange(event.target.value)}
                        value={field.state.value}
                      >
                        <option value="">Server default (America/New_York)</option>
                        {field.state.value && !allTimezones().includes(field.state.value) ? (
                          <option value={field.state.value}>{field.state.value}</option>
                        ) : null}
                        {allTimezones().map((tz) => (
                          <option key={tz} value={tz}>
                            {tz.replace(/_/g, ' ')}
                          </option>
                        ))}
                      </select>
                    )}
                  </form.Field>
                </label>
                <label className="field">
                  <span>Voice</span>
                  <form.Field name="voice">
                    {(field) => (
                      <select
                        disabled={formDisabled}
                        onChange={(event) => field.handleChange(event.target.value)}
                        value={field.state.value}
                      >
                        <option value="">Server default (af_heart)</option>
                        {field.state.value &&
                        !KOKORO_VOICES.some((v) => v.id === field.state.value) ? (
                          <option value={field.state.value}>{field.state.value}</option>
                        ) : null}
                        {KOKORO_VOICES.map((v) => (
                          <option key={v.id} value={v.id}>
                            {v.label}
                          </option>
                        ))}
                      </select>
                    )}
                  </form.Field>
                </label>
                <label className="field">
                  <form.Field name="speechRate">
                    {(field) => (
                      <>
                        <span>
                          Speech rate · <strong>{Number(field.state.value || 0.85).toFixed(2)}×</strong>
                        </span>
                        <input
                          disabled={formDisabled}
                          max="1.5"
                          min="0.5"
                          onChange={(event) => field.handleChange(event.target.value)}
                          step="0.05"
                          type="range"
                          value={field.state.value || '0.85'}
                        />
                      </>
                    )}
                  </form.Field>
                </label>
              </div>

              <label className="field">
                <span>Context (JSON, given to the prompt as personal data)</span>
                <form.Field name="contextJson">
                  {(field) => (
                    <textarea
                      className="settings-textarea settings-textarea-code"
                      disabled={formDisabled}
                      onChange={(event) => field.handleChange(event.target.value)}
                      placeholder='{"likes": "gardening"}'
                      rows={4}
                      value={field.state.value}
                    />
                  )}
                </form.Field>
              </label>

              {selectedProfile && !isCreatingProfile ? (
                <p className="form-note">
                  {selectedProfile.active ? 'Active' : 'Deactivated (operator action)'} · created{' '}
                  {formatTimestamp(selectedProfile.created_at)} · changes reach the device on its
                  next connect
                </p>
              ) : null}
            </>
          )}

          {selectedProfile && !isCreatingProfile ? (
            <>
              <div className="panel-divider" />

              <div className="panel-header">
                <h3>Linked accounts</h3>
              </div>
              {links.length ? (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Account</th>
                      <th>Role</th>
                      <th>Linked</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {links.map((link) => (
                      <tr key={link.account_id}>
                        <td>
                          {accountById.get(link.account_id)?.email || `account ${link.account_id}`}
                        </td>
                        <td>
                          <span className="pill pill-muted">{link.role}</span>
                        </td>
                        <td>{formatTimestamp(link.created_at)}</td>
                        <td>
                          <button
                            className="secondary-button"
                            onClick={() => void handleUnlink(link.account_id)}
                            type="button"
                          >
                            Unlink
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="empty-state">No accounts are linked to this profile.</div>
              )}

              <div className="settings-grid">
                <label className="field">
                  <span>Account</span>
                  <select
                    value={linkAccountId}
                    onChange={(event) => setLinkAccountId(event.target.value)}
                  >
                    <option value="">Select an account</option>
                    {accounts
                      .filter((account) => !links.some((link) => link.account_id === account.id))
                      .map((account) => (
                        <option key={account.id} value={account.id}>
                          {account.email}
                        </option>
                      ))}
                  </select>
                </label>
                <label className="field">
                  <span>Role</span>
                  <select
                    value={linkRole}
                    onChange={(event) => setLinkRole(event.target.value as 'owner' | 'viewer')}
                  >
                    <option value="owner">owner</option>
                    <option value="viewer">viewer</option>
                  </select>
                </label>
                <label className="field">
                  <span>&nbsp;</span>
                  <button
                    className="send-button"
                    disabled={!linkAccountId}
                    onClick={() => void handleLink()}
                    type="button"
                  >
                    Link account
                  </button>
                </label>
              </div>

              <div className="panel-divider" />

              <div className="panel-header">
                <h3>Device tokens</h3>
                <span>The raw token is shown once at issuance and is not recoverable.</span>
              </div>

              {issuedToken ? (
                <div className="token-reveal">
                  <strong>
                    Device token for “{issuedToken.label}” — copy it now, it will not be shown
                    again:
                  </strong>
                  <code>{issuedToken.token}</code>
                  <button
                    className="secondary-button"
                    onClick={() => void navigator.clipboard?.writeText(issuedToken.token)}
                    type="button"
                  >
                    Copy
                  </button>
                </div>
              ) : null}

              {tokens.length ? (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Label</th>
                      <th>Last used</th>
                      <th>Status</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {tokens.map((token) => (
                      <tr key={token.id}>
                        <td>{token.id}</td>
                        <td>{token.label || '—'}</td>
                        <td>{formatTimestamp(token.last_used_at)}</td>
                        <td>
                          {token.revoked_at ? (
                            <span className="pill pill-closed">revoked</span>
                          ) : (
                            <span className="pill pill-muted">active</span>
                          )}
                        </td>
                        <td>
                          {!token.revoked_at ? (
                            <button
                              className="secondary-button"
                              onClick={() => void handleRevokeToken(token.id)}
                              type="button"
                            >
                              Revoke
                            </button>
                          ) : null}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="empty-state">No device tokens for this profile yet.</div>
              )}

              <div className="settings-grid">
                <label className="field">
                  <span>New token label</span>
                  <input
                    onChange={(event) => setTokenLabel(event.target.value)}
                    placeholder="Tab A9 living room"
                    type="text"
                    value={tokenLabel}
                  />
                </label>
                <label className="field">
                  <span>&nbsp;</span>
                  <button
                    className="send-button"
                    disabled={!tokenLabel.trim()}
                    onClick={() => void handleIssueToken()}
                    type="button"
                  >
                    Issue device token
                  </button>
                </label>
              </div>
            </>
          ) : null}
        </section>

        <section className="timeline-panel insights-panel">
          <div className="panel-header">
            <h3>Accounts</h3>
            <span>Dashboard logins for care partners and the research team</span>
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
              disabled={
                isCreatingAccount ||
                !newAccountEmail.trim() ||
                !newAccountName.trim() ||
                newAccountPassword.length < 8
              }
              onClick={() => void handleCreateAccount()}
              type="button"
            >
              {isCreatingAccount ? 'Creating…' : 'Create account'}
            </button>
          </div>
        </section>
      </div>
    </AppShell>
  )
}
