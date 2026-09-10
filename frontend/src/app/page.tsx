'use client'

import { useEffect, useMemo, useState } from 'react'
import AppShell from '../components/AppShell'
import {
  fetchProfiles,
  fetchSessions,
  fetchTurns,
  isSessionExpired,
  type Profile,
  type SessionSummary,
  type Turn
} from '../lib/api'
import { useAuthGuard } from '../lib/useAuthGuard'

const SELECTED_PROFILE_STORAGE_KEY = 'robin-selected-profile-id'
const SESSIONS_PAGE_SIZE = 50

function formatTime(value: string | null | undefined) {
  if (!value) return 'No activity yet'

  return new Intl.DateTimeFormat([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit'
  }).format(new Date(value))
}

function eventClass(turn: Turn) {
  if (turn.source === 'proactive') return 'event event-chime'
  if (turn.role === 'assistant') return 'event event-assistant'
  return 'event event-user'
}

function roleLabel(turn: Turn) {
  if (turn.role === 'assistant') return 'Robin'
  if (turn.role === 'user') return 'User'
  return turn.role
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

export default function ChatsPage() {
  const { user, isChecking } = useAuthGuard()
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfileId, setSelectedProfileId] = useState<number | null>(null)
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [hasMoreSessions, setHasMoreSessions] = useState(false)
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null)
  const [turns, setTurns] = useState<Turn[]>([])
  const [isLoadingSessions, setIsLoadingSessions] = useState(false)
  const [isLoadingTurns, setIsLoadingTurns] = useState(false)
  const [roleFilter, setRoleFilter] = useState('all')
  const [sourceFilter, setSourceFilter] = useState('all')
  const [errorMessage, setErrorMessage] = useState('')

  const selectedProfile = profiles.find((profile) => profile.id === selectedProfileId) || null
  const selectedSession = sessions.find((session) => session.session_id === selectedSessionId) || null

  function handleRequestError(error: unknown, fallbackMessage: string) {
    if (isSessionExpired(error)) return
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

  async function loadSessions(profileId: number, before?: string) {
    setIsLoadingSessions(true)
    setErrorMessage('')
    try {
      const response = await fetchSessions(profileId, {
        limit: SESSIONS_PAGE_SIZE,
        ...(before ? { before } : {})
      })
      setSessions((current) => (before ? [...current, ...response.sessions] : response.sessions))
      setHasMoreSessions(response.sessions.length === SESSIONS_PAGE_SIZE)
      if (!before) {
        setSelectedSessionId(response.sessions[0]?.session_id ?? null)
      }
    } catch (error) {
      handleRequestError(error, 'Failed to load conversations.')
    } finally {
      setIsLoadingSessions(false)
    }
  }

  async function loadTurns(profileId: number, sessionId: string) {
    setIsLoadingTurns(true)
    setErrorMessage('')
    try {
      const response = await fetchTurns(profileId, { session_id: sessionId, limit: 500 })
      // The endpoint returns newest first; a conversation reads oldest first.
      setTurns([...response.turns].reverse())
    } catch (error) {
      handleRequestError(error, 'Failed to load the conversation.')
    } finally {
      setIsLoadingTurns(false)
    }
  }

  useEffect(() => {
    if (isChecking || !user) return
    void loadProfiles()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isChecking, user])

  useEffect(() => {
    setSessions([])
    setSelectedSessionId(null)
    setTurns([])
    if (!selectedProfileId) return
    void loadSessions(selectedProfileId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProfileId])

  useEffect(() => {
    setTurns([])
    if (!selectedProfileId || !selectedSessionId) return
    void loadTurns(selectedProfileId, selectedSessionId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSessionId])

  useEffect(() => {
    if (!canUseStorage()) return
    if (selectedProfileId == null) {
      window.localStorage.removeItem(SELECTED_PROFILE_STORAGE_KEY)
      return
    }
    window.localStorage.setItem(SELECTED_PROFILE_STORAGE_KEY, String(selectedProfileId))
  }, [selectedProfileId])

  const visibleTurns = useMemo(
    () =>
      turns.filter(
        (turn) =>
          (roleFilter === 'all' || turn.role === roleFilter) &&
          (sourceFilter === 'all' || turn.source === sourceFilter)
      ),
    [turns, roleFilter, sourceFilter]
  )

  if (isChecking) {
    return (
      <div className="auth-shell">
        <div className="auth-card auth-card-compact">
          <p className="eyebrow">Authentication</p>
          <h1>Checking session</h1>
          <p className="auth-copy">Validating your login before loading conversations.</p>
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
            <p className="eyebrow">Chats</p>
            <h2>{selectedProfile?.display_name || 'Select a profile'}</h2>
            <p>Every persisted utterance, grouped by conversation session.</p>
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

            <div className="hero-stats">
              <div className="stat-card">
                <span>Conversations</span>
                <strong>
                  {isLoadingSessions ? '…' : `${sessions.length}${hasMoreSessions ? '+' : ''}`}
                </strong>
              </div>
              <div className="stat-card">
                <span>Last activity</span>
                <strong>{formatTime(sessions[0]?.last_at ?? null)}</strong>
              </div>
            </div>
          </div>
        </div>
      </header>

      {errorMessage ? <section className="error-banner">{errorMessage}</section> : null}

      <section className="content-grid">
        <div className="timeline-panel">
          <div className="panel-header">
            <h3>Conversations</h3>
            <span>Newest first</span>
          </div>

          <div className="timeline">
            {!sessions.length && !isLoadingSessions ? (
              <div className="empty-state">No conversations recorded for this profile yet.</div>
            ) : (
              sessions.map((session) => (
                <button
                  key={session.session_id}
                  className={
                    session.session_id === selectedSessionId
                      ? 'patient-card patient-row nav-link-active'
                      : 'patient-card patient-row'
                  }
                  onClick={() => setSelectedSessionId(session.session_id)}
                  type="button"
                >
                  <div>
                    <strong>{formatTime(session.started_at)}</strong>
                    <p>
                      {session.turn_count} turn{session.turn_count === 1 ? '' : 's'} · ended{' '}
                      {formatTime(session.last_at)}
                    </p>
                  </div>
                  <div className="session-meta">
                    {session.sources.map((source) => (
                      <span key={source} className="pill pill-muted">
                        {source}
                      </span>
                    ))}
                  </div>
                </button>
              ))
            )}
          </div>

          {hasMoreSessions ? (
            <button
              className="secondary-button timeline-load-more"
              disabled={isLoadingSessions}
              onClick={() =>
                selectedProfileId &&
                void loadSessions(selectedProfileId, sessions[sessions.length - 1]?.last_at)
              }
              type="button"
            >
              {isLoadingSessions ? 'Loading…' : 'Load older conversations'}
            </button>
          ) : null}
        </div>

        <div className="timeline-panel">
          <div className="panel-header">
            <h3>
              {selectedSession
                ? `Conversation · ${formatTime(selectedSession.started_at)}`
                : 'Conversation'}
            </h3>
            <div className="inline-actions">
              <label className="field field-compact">
                <span>Role</span>
                <select value={roleFilter} onChange={(event) => setRoleFilter(event.target.value)}>
                  <option value="all">All</option>
                  <option value="user">User</option>
                  <option value="assistant">Robin</option>
                </select>
              </label>
              <label className="field field-compact">
                <span>Source</span>
                <select
                  value={sourceFilter}
                  onChange={(event) => setSourceFilter(event.target.value)}
                >
                  <option value="all">All</option>
                  <option value="voice">Voice</option>
                  <option value="proactive">Proactive</option>
                  <option value="tool">Tool</option>
                </select>
              </label>
            </div>
          </div>

          {!selectedSessionId ? (
            <div className="empty-state">Select a conversation.</div>
          ) : isLoadingTurns ? (
            <div className="empty-state">Loading conversation…</div>
          ) : !visibleTurns.length ? (
            <div className="empty-state">No turns match the current filters.</div>
          ) : (
            <div className="session-events">
              {visibleTurns.map((turn) => (
                <article key={turn.id} className={eventClass(turn)}>
                  <div className="event-meta">
                    <span className="pill">{roleLabel(turn)}</span>
                    <span className="pill pill-muted">{turn.source}</span>
                    {turn.latency_ms != null ? (
                      <span className="pill pill-muted">{turn.latency_ms} ms</span>
                    ) : null}
                    <time>{formatTime(turn.created_at)}</time>
                  </div>

                  <p className="event-body">{turn.content}</p>
                </article>
              ))}
            </div>
          )}
        </div>
      </section>
    </AppShell>
  )
}
