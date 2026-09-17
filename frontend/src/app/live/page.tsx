'use client'

/**
 * Live voice session — the browser diagnostic for the tablet client, sharing the
 * dashboard's login and profile picker. The WebSocket speaks the exact tablet protocol
 * over the same-origin /api proxy (the Next dev server forwards WS upgrades); see
 * useVoiceSession.
 *
 * Auth is deliberately invisible: the voice socket requires a device token (dashboard
 * logins are rejected), so the page mints a throwaway one per visit — labeled
 * "diagnostic: …", which is what keeps these sessions out of Chats/Activities by
 * default (robin/api/profiles.py) — and revokes it again on the way out. No caching,
 * nothing to manage. Token issuance is admin-only, so the page is too.
 */
import { useEffect, useState } from 'react'
import AppShell from '../../components/AppShell'
import {
  fetchHealth,
  fetchProfiles,
  isSessionExpired,
  issueDeviceToken,
  revokeToken,
  voiceWsUrl,
  type Profile
} from '../../lib/api'
import { useAuthGuard } from '../../lib/useAuthGuard'
import { useVoiceSession, type LiveTurn } from '../../lib/useVoiceSession'

const SELECTED_PROFILE_STORAGE_KEY = 'robin-selected-profile-id'

function getStoredSelectedProfileId() {
  if (typeof window === 'undefined') return null
  const rawValue = window.localStorage.getItem(SELECTED_PROFILE_STORAGE_KEY)
  if (!rawValue) return null
  const parsedValue = Number(rawValue)
  return Number.isFinite(parsedValue) ? parsedValue : null
}

export default function LivePage() {
  const { user, isChecking } = useAuthGuard({ requireAdmin: true })
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfileId, setSelectedProfileId] = useState<number | null>(null)
  const [deviceToken, setDeviceToken] = useState('')
  const [turnMode, setTurnMode] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState('')

  const selectedProfile = profiles.find((profile) => profile.id === selectedProfileId) || null

  function handleRequestError(error: unknown, fallbackMessage: string) {
    if (isSessionExpired(error)) return
    setErrorMessage(error instanceof Error ? error.message : fallbackMessage)
  }

  useEffect(() => {
    if (isChecking || !user) return
    void (async () => {
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
      try {
        setTurnMode((await fetchHealth()).turn_mode)
      } catch {
        setErrorMessage('Voice server unreachable — is it running on port 8000?')
      }
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isChecking, user])

  useEffect(() => {
    if (selectedProfileId == null) return
    window.localStorage.setItem(SELECTED_PROFILE_STORAGE_KEY, String(selectedProfileId))
  }, [selectedProfileId])

  // One throwaway device token per profile visit: minted here, revoked in cleanup
  // (profile switch, navigation away, or a dev-mode remount). Best-effort revoke —
  // an orphaned diagnostic token grants nothing new and is filtered from the data
  // views regardless.
  useEffect(() => {
    if (!user?.is_admin || selectedProfileId == null) return
    let cancelled = false
    let issuedTokenId: number | null = null
    setDeviceToken('')
    setErrorMessage('')
    void (async () => {
      try {
        const issued = await issueDeviceToken(
          selectedProfileId, `diagnostic: browser — ${user.email}`)
        issuedTokenId = issued.token_id
        if (!cancelled) setDeviceToken(issued.token)
      } catch (error) {
        if (!cancelled) handleRequestError(error, 'Failed to issue a session token.')
      }
    })()
    return () => {
      cancelled = true
      if (issuedTokenId != null) void revokeToken(issuedTokenId).catch(() => undefined)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProfileId, user?.is_admin])

  if (isChecking) {
    return (
      <div className="auth-shell">
        <div className="auth-card auth-card-compact">
          <p className="eyebrow">Authentication</p>
          <h1>Checking session</h1>
          <p className="auth-copy">Validating your login before starting a live session.</p>
        </div>
      </div>
    )
  }

  if (!user) return null

  return (
    <AppShell user={user}>
      <header className="hero">
        <div className="hero-main">
          <div>
            <p className="eyebrow">Live</p>
            <h2>{selectedProfile?.display_name || 'Select a profile'}</h2>
            <p>
              Talk to Robin from the browser over the same wire protocol as the tablet.
              Sessions are recorded as diagnostics and hidden from Chats and Activities
              by default.
            </p>
          </div>

          <div className="hero-filters">
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
          </div>
        </div>
      </header>

      {errorMessage ? <section className="error-banner">{errorMessage}</section> : null}

      {!selectedProfile ? (
        <section className="timeline-panel">
          <div className="empty-state">Select a profile to start a live session.</div>
        </section>
      ) : turnMode && turnMode !== 'tap' ? (
        <section className="timeline-panel">
          <div className="empty-state">
            The server is running turn_mode=&quot;{turnMode}&quot;; only tap mode is supported
            here. Use the plain page at the server origin for hold-to-talk.
          </div>
        </section>
      ) : !deviceToken ? (
        <section className="timeline-panel">
          <div className="empty-state">Starting a session…</div>
        </section>
      ) : (
        <LiveSessionPanel
          key={`${selectedProfile.id}:${deviceToken}`}
          deviceToken={deviceToken}
          onAuthRejected={() => {
            setDeviceToken('')
            setErrorMessage(
              'The session token was rejected by the voice server — is the profile active?')
          }}
        />
      )}
    </AppShell>
  )
}

function LiveSessionPanel({
  deviceToken,
  onAuthRejected
}: {
  deviceToken: string
  onAuthRejected: () => void
}) {
  const session = useVoiceSession(voiceWsUrl(), deviceToken, onAuthRejected)
  const { tap } = session

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.code === 'Space' && !event.repeat &&
          !(event.target instanceof HTMLInputElement) &&
          !(event.target instanceof HTMLTextAreaElement)) {
        event.preventDefault()
        tap()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [tap])

  return (
    <section className="timeline-panel live-panel">
      <div className="panel-header">
        <h3>Live conversation</h3>
        <span>{session.isConnected ? 'connected' : session.status}</span>
      </div>

      <div className="session-events live-transcript">
        {!session.turns.length ? (
          <div className="empty-state">
            Tap once (or press Space) and speak. The conversation then keeps going turn by
            turn — stay quiet when it&apos;s your turn and it drops back to tap-to-talk.
          </div>
        ) : (
          session.turns.map((turn: LiveTurn) => (
            <article
              key={turn.id}
              className={turn.role === 'assistant' ? 'event event-assistant' : 'event event-user'}
            >
              <div className="event-meta">
                <span className="pill">{turn.role === 'assistant' ? 'Robin' : 'You'}</span>
              </div>
              <p className="event-body">{turn.text || '…'}</p>
            </article>
          ))
        )}
      </div>

      {/* VAD row: state chip + live speech-probability meter (essential for tuning
          hangover by feel — carried over from tap_index.html). */}
      <div className="live-vad-row">
        <span className={`live-vad-state live-vad-${session.vadState.toLowerCase()}`}>
          {session.vadState.toLowerCase()}
        </span>
        <div className="live-meter-track">
          <div className="live-meter-fill" style={{ width: `${Math.round(session.prob * 100)}%` }} />
        </div>
        <span className="live-vad-prob">{session.prob.toFixed(2)}</span>
      </div>

      <div className="live-status">{session.status}</div>

      {session.isConnected ? (
        <button className="live-talk-button" disabled={!session.canTap} onClick={tap} type="button">
          {session.canTap ? 'Tap to talk' : 'listening…'}
        </button>
      ) : (
        <button className="live-talk-button" onClick={session.reconnect} type="button">
          Reconnect
        </button>
      )}
    </section>
  )
}
