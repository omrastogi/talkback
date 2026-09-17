'use client'

import { useEffect, useMemo, useState } from 'react'
import AppShell from '../../components/AppShell'
import {
  fetchActivity,
  fetchProfiles,
  isSessionExpired,
  type Activity,
  type Profile
} from '../../lib/api'
import { useAuthGuard } from '../../lib/useAuthGuard'

const SELECTED_PROFILE_STORAGE_KEY = 'robin-selected-profile-id'

type ChartTab = 'sessions' | 'turns'

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

function formatDay(day: string) {
  return new Intl.DateTimeFormat([], { month: 'short', day: 'numeric' }).format(
    new Date(`${day}T00:00:00`)
  )
}

function formatTimestamp(value: string | null) {
  if (!value) return 'Never'
  return new Intl.DateTimeFormat([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit'
  }).format(new Date(value))
}

function toCsvValue(value: string | number | boolean) {
  const stringValue = String(value ?? '')
  if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
    return `"${stringValue.replace(/"/g, '""')}"`
  }
  return stringValue
}

export default function ActivitiesPage() {
  const { user, isChecking } = useAuthGuard()
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfileId, setSelectedProfileId] = useState<number | null>(null)
  const [activity, setActivity] = useState<Activity | null>(null)
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [activeTab, setActiveTab] = useState<ChartTab>('sessions')
  const [includeDiagnostic, setIncludeDiagnostic] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')

  const selectedProfile = profiles.find((profile) => profile.id === selectedProfileId) || null

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

  async function loadActivity(profileId: number) {
    setIsLoading(true)
    setErrorMessage('')
    try {
      setActivity(
        await fetchActivity(profileId, {
          include_diagnostic: includeDiagnostic,
          ...(dateFrom ? { date_from: dateFrom } : {}),
          ...(dateTo ? { date_to: dateTo } : {})
        })
      )
    } catch (error) {
      handleRequestError(error, 'Failed to load activity.')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (isChecking || !user) return
    void loadProfiles()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isChecking, user])

  useEffect(() => {
    setActivity(null)
    if (!selectedProfileId) return
    void loadActivity(selectedProfileId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProfileId, dateFrom, dateTo, includeDiagnostic])

  useEffect(() => {
    if (!canUseStorage()) return
    if (selectedProfileId == null) {
      window.localStorage.removeItem(SELECTED_PROFILE_STORAGE_KEY)
      return
    }
    window.localStorage.setItem(SELECTED_PROFILE_STORAGE_KEY, String(selectedProfileId))
  }, [selectedProfileId])

  const totals = useMemo(() => {
    const days = activity?.days ?? []
    const totalTurns = days.reduce((sum, d) => sum + d.user_turns + d.assistant_turns, 0)
    const latencies = days.filter((d) => d.avg_latency_ms != null)
    return {
      sessions: days.reduce((sum, d) => sum + d.sessions, 0),
      turns: totalTurns,
      userTurns: days.reduce((sum, d) => sum + d.user_turns, 0),
      proactiveTurns: days.reduce((sum, d) => sum + d.proactive_turns, 0),
      avgLatencyMs: latencies.length
        ? Math.round(
            latencies.reduce((sum, d) => sum + (d.avg_latency_ms ?? 0), 0) / latencies.length
          )
        : null
    }
  }, [activity])

  const maxBarValue = useMemo(() => {
    const days = activity?.days ?? []
    const values = days.map((d) =>
      activeTab === 'sessions' ? d.sessions : d.user_turns + d.assistant_turns
    )
    return Math.max(1, ...values)
  }, [activity, activeTab])

  function handleClearRange() {
    setDateFrom('')
    setDateTo('')
  }

  function handleDownloadCsv() {
    if (!activity?.days.length) return
    const header = [
      'day',
      'sessions',
      'user_turns',
      'assistant_turns',
      'proactive_turns',
      'voice_turns',
      'avg_latency_ms'
    ]
    const rows = activity.days.map((d) =>
      [d.day, d.sessions, d.user_turns, d.assistant_turns, d.proactive_turns, d.voice_turns,
       d.avg_latency_ms ?? ''].map(toCsvValue).join(',')
    )
    const blob = new Blob([[header.join(','), ...rows].join('\n')], {
      type: 'text/csv;charset=utf-8'
    })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `robin-activity-profile-${selectedProfileId}.csv`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  if (isChecking) {
    return (
      <div className="auth-shell">
        <div className="auth-card auth-card-compact">
          <p className="eyebrow">Authentication</p>
          <h1>Checking session</h1>
          <p className="auth-copy">Validating your login before loading activity.</p>
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
            <p className="eyebrow">Activities</p>
            <h2>{selectedProfile?.display_name || 'Select a profile'}</h2>
            <p>
              Daily usage derived from the conversation history
              {activity ? ` · days in ${activity.timezone}` : ''}. Defaults to the last 30 days.
            </p>
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
            <label className="field field-compact">
              <span>From</span>
              <input
                onChange={(event) => setDateFrom(event.target.value)}
                type="date"
                value={dateFrom}
              />
            </label>
            <label className="field field-compact">
              <span>To</span>
              <input
                onChange={(event) => setDateTo(event.target.value)}
                type="date"
                value={dateTo}
              />
            </label>
            <label className="auth-checkbox">
              <input
                checked={includeDiagnostic}
                onChange={(event) => setIncludeDiagnostic(event.target.checked)}
                type="checkbox"
              />
              <span>Show diagnostics</span>
            </label>
            <div className="filter-actions">
              <button className="filter-clear-button" onClick={handleClearRange} type="button">
                Clear range
              </button>
              <button
                className="secondary-button"
                disabled={!activity?.days.length}
                onClick={handleDownloadCsv}
                type="button"
              >
                Download CSV
              </button>
            </div>
          </div>
        </div>
      </header>

      {errorMessage ? <section className="error-banner">{errorMessage}</section> : null}

      <div className="patients-layout">
        <section className="timeline-panel insights-panel">
          {!selectedProfileId ? (
            <div className="empty-state">Select a profile.</div>
          ) : isLoading ? (
            <div className="empty-state">Loading activity…</div>
          ) : (
            <>
              <div className="insight-stat-grid">
                <article className="insight-stat-card">
                  <span>Conversations</span>
                  <strong>{totals.sessions}</strong>
                </article>
                <article className="insight-stat-card">
                  <span>Total turns</span>
                  <strong>{totals.turns}</strong>
                </article>
                <article className="insight-stat-card">
                  <span>User turns</span>
                  <strong>{totals.userTurns}</strong>
                </article>
                <article className="insight-stat-card">
                  <span>Proactive turns</span>
                  <strong>{totals.proactiveTurns}</strong>
                </article>
                <article className="insight-stat-card">
                  <span>Avg latency</span>
                  <strong>{totals.avgLatencyMs != null ? `${totals.avgLatencyMs} ms` : '—'}</strong>
                </article>
                <article className="insight-stat-card">
                  <span>Last active</span>
                  <strong>{formatTimestamp(activity?.last_active_at ?? null)}</strong>
                </article>
              </div>

              <div className="panel-divider" />

              <div className="insights-tabs">
                <button
                  className={activeTab === 'sessions' ? 'tab-button tab-button-active' : 'tab-button'}
                  onClick={() => setActiveTab('sessions')}
                  type="button"
                >
                  Conversations per day
                </button>
                <button
                  className={activeTab === 'turns' ? 'tab-button tab-button-active' : 'tab-button'}
                  onClick={() => setActiveTab('turns')}
                  type="button"
                >
                  Turns per day
                </button>
              </div>

              {!activity?.days.length ? (
                <div className="empty-state">No activity in this date range.</div>
              ) : (
                <div className="bar-chart">
                  {activity.days.map((day) => {
                    const value =
                      activeTab === 'sessions' ? day.sessions : day.user_turns + day.assistant_turns
                    return (
                      <div className="bar-row" key={day.day}>
                        <span>{formatDay(day.day)}</span>
                        <div className="bar-track">
                          {activeTab === 'turns' ? (
                            <>
                              <div
                                className="bar-fill"
                                style={{ width: `${(day.user_turns / maxBarValue) * 100}%` }}
                                title={`${day.user_turns} user turns`}
                              />
                              <div
                                className="bar-fill bar-fill-muted"
                                style={{ width: `${(day.assistant_turns / maxBarValue) * 100}%` }}
                                title={`${day.assistant_turns} Robin turns`}
                              />
                            </>
                          ) : (
                            <div
                              className="bar-fill"
                              style={{ width: `${(value / maxBarValue) * 100}%` }}
                            />
                          )}
                        </div>
                        <strong>
                          {activeTab === 'turns'
                            ? `${day.user_turns} user · ${day.assistant_turns} Robin${
                                day.proactive_turns ? ` · ${day.proactive_turns} proactive` : ''
                              }`
                            : value}
                        </strong>
                      </div>
                    )
                  })}
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </AppShell>
  )
}
