'use client'

import { useEffect, useRef, useState } from 'react'
import { useForm, useStore } from '@tanstack/react-form'
import AppShell from '../../components/AppShell'
import {
  createProfile,
  fetchAccounts,
  fetchProfileLinks,
  fetchProfiles,
  fetchTokens,
  fetchVoicePreview,
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

// All English Kokoro-82M voices (weights are pre-downloaded on the server). Prefix
// decodes as a=American/b=British + f=female/m=male.
const KOKORO_VOICES = [
  { id: 'af_heart', label: 'Heart — American female (default)' },
  { id: 'af_alloy', label: 'Alloy — American female' },
  { id: 'af_aoede', label: 'Aoede — American female' },
  { id: 'af_bella', label: 'Bella — American female' },
  { id: 'af_jessica', label: 'Jessica — American female' },
  { id: 'af_kore', label: 'Kore — American female' },
  { id: 'af_nicole', label: 'Nicole — American female' },
  { id: 'af_nova', label: 'Nova — American female' },
  { id: 'af_river', label: 'River — American female' },
  { id: 'af_sarah', label: 'Sarah — American female' },
  { id: 'af_sky', label: 'Sky — American female' },
  { id: 'am_adam', label: 'Adam — American male' },
  { id: 'am_echo', label: 'Echo — American male' },
  { id: 'am_eric', label: 'Eric — American male' },
  { id: 'am_fenrir', label: 'Fenrir — American male' },
  { id: 'am_liam', label: 'Liam — American male' },
  { id: 'am_michael', label: 'Michael — American male' },
  { id: 'am_onyx', label: 'Onyx — American male' },
  { id: 'am_puck', label: 'Puck — American male' },
  { id: 'am_santa', label: 'Santa — American male' },
  { id: 'bf_alice', label: 'Alice — British female' },
  { id: 'bf_emma', label: 'Emma — British female' },
  { id: 'bf_isabella', label: 'Isabella — British female' },
  { id: 'bf_lily', label: 'Lily — British female' },
  { id: 'bm_daniel', label: 'Daniel — British male' },
  { id: 'bm_fable', label: 'Fable — British male' },
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

  // Link + token forms. Accounts themselves are managed on the Accounts page; the list
  // is only needed here to show emails and fill the link dropdown.
  const [linkAccountId, setLinkAccountId] = useState('')
  const [linkRole, setLinkRole] = useState<'owner' | 'viewer'>('owner')
  const [tokenLabel, setTokenLabel] = useState('')

  // Voice preview panel. Blob URLs are cached per voice for the page's lifetime.
  const [isVoicePanelOpen, setIsVoicePanelOpen] = useState(false)
  const [playingVoice, setPlayingVoice] = useState<string | null>(null)
  const [loadingVoice, setLoadingVoice] = useState<string | null>(null)
  const previewUrlsRef = useRef(new Map<string, string>())
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const selectedProfile = profiles.find((profile) => profile.id === selectedProfileId) || null

  const form = useForm({
    defaultValues: EMPTY_PROFILE_FORM,

    onSubmit: async ({ value, formApi }) => {
      if (!value.displayName.trim()) {
        setErrorMessage('Display name is required.')
        return
      }
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
            // Empty means "leave unchanged" — the backend rejects empty strings.
            ...(value.timezone.trim() ? { timezone: value.timezone.trim() } : {}),
            ...(value.voice.trim() ? { voice: value.voice.trim() } : {}),
            ...(speechRate !== null ? { speech_rate: speechRate } : {}),
            context
          })
          formApi.reset(value, { keepDefaultValues: true })
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

  function stopVoicePlayback() {
    audioRef.current?.pause()
    audioRef.current = null
    setPlayingVoice(null)
  }

  async function handlePlayVoice(voiceId: string) {
    if (playingVoice === voiceId) {
      stopVoicePlayback()
      return
    }
    stopVoicePlayback()
    try {
      let url = previewUrlsRef.current.get(voiceId)
      if (!url) {
        setLoadingVoice(voiceId)
        const blob = await fetchVoicePreview(voiceId)
        url = URL.createObjectURL(blob)
        previewUrlsRef.current.set(voiceId, url)
      }
      const audio = new Audio(url)
      audio.onended = () => setPlayingVoice((current) => (current === voiceId ? null : current))
      audioRef.current = audio
      setPlayingVoice(voiceId)
      await audio.play()
    } catch (error) {
      handleRequestError(error, `Could not play a preview of ${voiceId}.`)
      setPlayingVoice(null)
    } finally {
      setLoadingVoice(null)
    }
  }

  function handleStartCreateProfile() {
    setIsCreatingProfile(true)
    setIsEditingProfile(false)
    setErrorMessage('')
    form.reset(EMPTY_PROFILE_FORM, { keepDefaultValues: true })
  }

  function handleStartEditProfile() {
    if (!selectedProfile) return
    setIsEditingProfile(true)
    setIsCreatingProfile(false)
    setErrorMessage('')
    // keepDefaultValues on every reset: without it, reset(values) rewrites the form's
    // defaultValues, and useForm's per-render options sync then clobbers the values
    // back to EMPTY_PROFILE_FORM on the next render (the form is untouched post-reset).
    form.reset(buildProfileForm(selectedProfile), { keepDefaultValues: true })
  }

  function handleCancelProfileForm() {
    setIsCreatingProfile(false)
    setIsEditingProfile(false)
    setErrorMessage('')
    form.reset(selectedProfile ? buildProfileForm(selectedProfile) : EMPTY_PROFILE_FORM, {
      keepDefaultValues: true
    })
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
      form.reset(EMPTY_PROFILE_FORM, { keepDefaultValues: true })
      return
    }
    void loadLinksAndTokens(selectedProfileId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProfileId])

  useEffect(() => {
    if (isCreatingProfile || isEditingProfile) return
    // Never clobber edits that are still in flight.
    if (!form.state.isPristine) return
    form.reset(selectedProfile ? buildProfileForm(selectedProfile) : EMPTY_PROFILE_FORM, {
      keepDefaultValues: true
    })
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

  useEffect(() => {
    const urls = previewUrlsRef.current
    return () => {
      audioRef.current?.pause()
      urls.forEach((url) => URL.revokeObjectURL(url))
      urls.clear()
    }
  }, [])

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
            <p className="eyebrow">Profiles</p>
            <h2>{isCreatingProfile ? 'New profile' : selectedProfile?.display_name || 'Profiles'}</h2>
            <p>The people Robin talks to — their voice, timezone, device tokens, and who can see them.</p>
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
              <strong>{profiles.length}</strong> profiles
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
                  <div className="voice-select-row">
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
                    <button
                      className="secondary-button"
                      onClick={() => {
                        if (isVoicePanelOpen) stopVoicePlayback()
                        setIsVoicePanelOpen(!isVoicePanelOpen)
                      }}
                      type="button"
                    >
                      {isVoicePanelOpen ? 'Hide voices' : '🔊 Listen'}
                    </button>
                  </div>
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

              {isVoicePanelOpen ? (
                <div className="voice-preview-panel">
                  {KOKORO_VOICES.map((v) => (
                    <div className="voice-preview-row" key={v.id}>
                      <button
                        className="secondary-button"
                        disabled={loadingVoice !== null && loadingVoice !== v.id}
                        onClick={() => void handlePlayVoice(v.id)}
                        type="button"
                      >
                        {loadingVoice === v.id ? '…' : playingVoice === v.id ? '◼' : '▶'}
                      </button>
                      <span>{v.label}</span>
                    </div>
                  ))}
                  <p className="form-note">
                    The first play of a voice takes a few seconds while the server synthesizes
                    it (and downloads the voice if it isn't cached yet).
                  </p>
                </div>
              ) : null}

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
                <h3>Dashboard access</h3>
                <span>Accounts are created on the Accounts page</span>
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

      </div>
    </AppShell>
  )
}
