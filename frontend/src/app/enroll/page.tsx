'use client'

/**
 * Wake-word enrollment: record "Hey Robin" takes against a profile, review and prune
 * them, and see whether a personalized head is live. Recording happens client-side at
 * 16 kHz mono PCM16 (useWakeRecorder) — the trainer's exact input contract — and each
 * take uploads immediately.
 *
 * Training itself is deliberately not a dashboard button: it runs offline in the
 * oww-train repo against these stored clips (robin/README.md, "Wake-word
 * personalization"), and the tablet picks the trained head up over the voice socket on
 * its next connect.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import AppShell from '../../components/AppShell'
import {
  deleteWakeClip,
  fetchProfiles,
  fetchWakeClipAudio,
  fetchWakeClips,
  fetchWakeModel,
  isSessionExpired,
  trainWakeModel,
  uploadWakeClip,
  type Profile,
  type WakeClip,
  type WakeModelStatus
} from '../../lib/api'
import { useAuthGuard } from '../../lib/useAuthGuard'
import { useWakeRecorder } from '../../lib/useWakeRecorder'

const SELECTED_PROFILE_STORAGE_KEY = 'robin-selected-profile-id'
const TAKE_SECONDS = 2
const TARGET_POSITIVES = 12 // matches the "8-15 clips" enrollment design in oww-train
const MIN_POSITIVES = 1     // no forced enrollment size — zero would be nothing to train on

function getStoredSelectedProfileId() {
  if (typeof window === 'undefined') return null
  const rawValue = window.localStorage.getItem(SELECTED_PROFILE_STORAGE_KEY)
  if (!rawValue) return null
  const parsedValue = Number(rawValue)
  return Number.isFinite(parsedValue) ? parsedValue : null
}

function formatTimestamp(value: string) {
  return new Date(value).toLocaleString()
}

export default function EnrollPage() {
  const { user, isChecking } = useAuthGuard()
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfileId, setSelectedProfileId] = useState<number | null>(null)
  const [clips, setClips] = useState<WakeClip[]>([])
  const [model, setModel] = useState<WakeModelStatus | null>(null)
  const [statusMessage, setStatusMessage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const [playingClipId, setPlayingClipId] = useState<number | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const recorder = useWakeRecorder()

  const selectedProfile = profiles.find((profile) => profile.id === selectedProfileId) || null
  const positives = clips.filter((clip) => clip.label === 'positive')
  const negatives = clips.filter((clip) => clip.label === 'negative')

  function handleRequestError(error: unknown, fallbackMessage: string) {
    if (isSessionExpired(error)) return
    setErrorMessage(error instanceof Error ? error.message : fallbackMessage)
  }

  const reloadProfileData = useCallback(async (profileId: number) => {
    try {
      const [clipsResponse, modelResponse] = await Promise.all([
        fetchWakeClips(profileId),
        fetchWakeModel(profileId)
      ])
      setClips(clipsResponse.clips)
      setModel(modelResponse)
    } catch (error) {
      handleRequestError(error, 'Failed to load enrollment data.')
    }
  }, [])

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
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isChecking, user])

  useEffect(() => {
    if (selectedProfileId == null) return
    window.localStorage.setItem(SELECTED_PROFILE_STORAGE_KEY, String(selectedProfileId))
    setClips([])
    setModel(null)
    setErrorMessage('')
    setStatusMessage('')
    void reloadProfileData(selectedProfileId)
  }, [selectedProfileId, reloadProfileData])

  async function handleRecord() {
    if (selectedProfileId == null || recorder.isRecording || isUploading) return
    setErrorMessage('')
    setStatusMessage(`Recording — say “Hey Robin” now (${TAKE_SECONDS}s)`)
    let wav: Blob
    try {
      wav = await recorder.record(TAKE_SECONDS)
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error)
      setStatusMessage('')
      setErrorMessage(detail.includes('microphone') ? detail : `microphone error: ${detail}`)
      return
    }
    setStatusMessage('Checking and saving the take…')
    setIsUploading(true)
    try {
      await uploadWakeClip(selectedProfileId, wav)
      setStatusMessage('Take saved.')
      await reloadProfileData(selectedProfileId)
    } catch (error) {
      setStatusMessage('')
      handleRequestError(error, 'Failed to upload the take.')
    } finally {
      setIsUploading(false)
    }
  }

  async function handleRegister() {
    if (selectedProfileId == null) return
    setErrorMessage('')
    try {
      await trainWakeModel(selectedProfileId)
      setModel(await fetchWakeModel(selectedProfileId))
    } catch (error) {
      handleRequestError(error, 'Failed to start training.')
    }
  }

  // While a training run is live, poll its status; the run flips clips_changed off and
  // publishes the new head when it lands, which is what re-locks the Register button.
  const trainingState = model?.training?.state
  useEffect(() => {
    if (selectedProfileId == null || trainingState !== 'running') return
    const timer = window.setInterval(() => {
      void fetchWakeModel(selectedProfileId)
        .then(setModel)
        .catch(() => undefined)
    }, 5000)
    return () => window.clearInterval(timer)
  }, [selectedProfileId, trainingState])

  async function handlePlay(clip: WakeClip) {
    if (selectedProfileId == null) return
    audioRef.current?.pause()
    setPlayingClipId(clip.id)
    try {
      const blob = await fetchWakeClipAudio(selectedProfileId, clip.id)
      const url = URL.createObjectURL(blob)
      const audio = new Audio(url)
      audioRef.current = audio
      audio.onended = () => {
        URL.revokeObjectURL(url)
        setPlayingClipId((current) => (current === clip.id ? null : current))
      }
      await audio.play()
    } catch (error) {
      setPlayingClipId(null)
      handleRequestError(error, 'Failed to play the take.')
    }
  }

  async function handleDelete(clip: WakeClip) {
    if (selectedProfileId == null) return
    try {
      await deleteWakeClip(selectedProfileId, clip.id)
      await reloadProfileData(selectedProfileId)
    } catch (error) {
      handleRequestError(error, 'Failed to delete the take.')
    }
  }

  useEffect(() => () => audioRef.current?.pause(), [])

  if (isChecking) {
    return (
      <div className="auth-shell">
        <div className="auth-card auth-card-compact">
          <p className="eyebrow">Authentication</p>
          <h1>Checking session</h1>
          <p className="auth-copy">Validating your login before opening enrollment.</p>
        </div>
      </div>
    )
  }

  if (!user) return null

  const canRecord = selectedProfile != null &&
    (selectedProfile.role === 'owner' || selectedProfile.role === 'admin')

  // Register lock: the server compares current take hashes against the ones the active
  // model was trained on. Same set -> locked; any recording or deletion -> unlocked.
  const isTraining = model?.training?.state === 'running'
  const canRegister = model != null && !isTraining &&
    model.clips_changed && model.positives >= MIN_POSITIVES
  const registerLabel = isTraining
    ? 'Registering voice…'
    : model != null && model.positives < MIN_POSITIVES
      ? 'Record a take to register'
      : model != null && !model.clips_changed && model.available
        ? 'Voice registered ✓'
        : 'Register Voice'

  return (
    <AppShell user={user}>
      <header className="hero">
        <div className="hero-main">
          <div>
            <p className="eyebrow">Wake word</p>
            <h2>{selectedProfile?.display_name || 'Select a profile'}</h2>
            <p>Personalize this profile&apos;s “Hey Robin” detector with recorded takes.</p>
          </div>

          <div className="hero-filters">
            {/* min-width: a long hero paragraph must squeeze the text, never this select */}
            <label className="field field-patient" style={{ minWidth: 220 }}>
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
          <div className="empty-state">Select a profile to record enrollment takes.</div>
        </section>
      ) : (
        <>
          <section className="timeline-panel">
            <div className="panel-header">
              <h3>Personalized model</h3>
              <span>
                {model == null
                  ? 'checking…'
                  : model.training?.state === 'running'
                    ? 'registering…'
                    : model.available
                      ? `active — base ${model.base_version}, threshold ${model.threshold?.toFixed(3)}`
                      : 'none — the tablet uses the shared base model'}
              </span>
            </div>
            {model?.available ? (
              <p className="event-body">
                Registered {model.created_at ? formatTimestamp(model.created_at) : '—'} · sha{' '}
                {model.sha256?.slice(0, 12)} · delivered to the tablet on its next connect.
              </p>
            ) : null}
            {model?.training?.state === 'failed' ? (
              <p className="event-body">Last registration failed: {model.training.detail}</p>
            ) : null}
            {model != null && canRecord ? (
              <>
                <button
                  className="live-talk-button"
                  disabled={!canRegister}
                  onClick={() => void handleRegister()}
                  type="button"
                >
                  {registerLabel}
                </button>
                {model.training?.state === 'running' ? (
                  <div className="live-status">
                    Registering this voice from the current takes — a few minutes; the
                    page updates by itself.
                  </div>
                ) : null}
              </>
            ) : null}
          </section>

          <section className="timeline-panel">
            <div className="panel-header">
              <h3>Record takes</h3>
              <span>
                {positives.length}/{TARGET_POSITIVES} “Hey Robin” takes
                {negatives.length ? ` · ${negatives.length} other-speech takes` : ''}
              </span>
            </div>

            <p className="event-body">
              Aim for {TARGET_POSITIVES}+ takes in the person&apos;s natural voice, at a normal
              distance from the microphone. Each take is checked as it uploads — a garbled
              one is rejected with a message and can simply be re-recorded. Registering
              builds a personalized model from these takes and it reaches the tablet
              automatically.
            </p>
            {!canRecord ? (
              <div className="empty-state">
                Your role on this profile is view-only; recording needs the owner role.
              </div>
            ) : (
              <>
                <button
                  className="live-talk-button"
                  disabled={recorder.isRecording || isUploading}
                  onClick={() => void handleRecord()}
                  type="button"
                >
                  {recorder.isRecording
                    ? '● recording…'
                    : isUploading
                      ? 'saving…'
                      : `Record ${TAKE_SECONDS}s take`}
                </button>
                <div className="live-status">{statusMessage}</div>
              </>
            )}
          </section>

          <section className="timeline-panel">
            <div className="panel-header">
              <h3>Stored takes</h3>
              <span>{clips.length} total</span>
            </div>
            {!clips.length ? (
              <div className="empty-state">
                No takes yet. Press record and say “Hey Robin” as you naturally would.
              </div>
            ) : (
              <div className="session-events">
                {clips.map((clip) => (
                  <article key={clip.id} className="event">
                    <div className="event-meta">
                      <span className="pill">
                        {clip.label === 'positive' ? 'Hey Robin' : 'other speech'}
                      </span>
                      <span>{clip.duration_s.toFixed(1)}s</span>
                      <span>{formatTimestamp(clip.created_at)}</span>
                    </div>
                    <div className="event-body">
                      <button
                        type="button"
                        onClick={() => void handlePlay(clip)}
                        disabled={playingClipId === clip.id}
                      >
                        {playingClipId === clip.id ? 'playing…' : 'Play'}
                      </button>{' '}
                      {canRecord ? (
                        <button type="button" onClick={() => void handleDelete(clip)}>
                          Delete
                        </button>
                      ) : null}
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        </>
      )}
    </AppShell>
  )
}
