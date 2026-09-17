'use client'

/**
 * Live voice session over the same /ws-stream protocol the tablet client uses —
 * a straight port of tap_index.html's tap-to-talk logic (VAD endpointing, gapless
 * reply playback, hands-free re-arm). No dashboard-specific frames: this page is a
 * diagnostic stand-in for the tablet, so the wire traffic must stay identical.
 *
 * All mutable audio/socket state lives in refs; React state only mirrors what the
 * page renders. Everything is torn down on unmount (Fast Refresh remounts during
 * development, and a leaked AudioContext keeps the mic hot).
 */
import { useCallback, useEffect, useRef, useState } from 'react'

export type VadState = 'IDLE' | 'ARMED' | 'SPEECH' | 'TRAILING'

export interface LiveTurn {
  id: number
  role: 'user' | 'assistant'
  text: string
}

// Same worklet as tap_index.html: forward each 128-sample float32 block to JS.
const WORKLET = `
class Recorder extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor('recorder', Recorder);
`

const WS_CLOSE_UNAUTHORIZED = 4401 // bad/revoked/non-device token (see robin/ws.py)

export interface VoiceSession {
  status: string
  vadState: VadState
  prob: number
  turns: LiveTurn[]
  canTap: boolean
  isConnected: boolean
  tap: () => void
  reconnect: () => void
}

export function useVoiceSession(wsUrl: string, deviceToken: string,
                                onAuthRejected: () => void): VoiceSession {
  const [status, setStatus] = useState('connecting…')
  const [vadState, setVadState] = useState<VadState>('IDLE')
  const [prob, setProb] = useState(0)
  const [turns, setTurns] = useState<LiveTurn[]>([])
  const [canTap, setCanTap] = useState(false)
  const [isConnected, setIsConnected] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const playCtxRef = useRef<AudioContext | null>(null)
  const nextStartRef = useRef(0)
  const capCtxRef = useRef<AudioContext | null>(null)
  const micStreamRef = useRef<MediaStream | null>(null)
  const micOpenRef = useRef(false)
  const clientStateRef = useRef<'IDLE' | 'ARMED'>('IDLE') // tap allowed only when IDLE
  const audioPromisesRef = useRef<Promise<void>[]>([])
  const rearmTimerRef = useRef<number | null>(null)
  const disposedRef = useRef(false)
  const turnIdRef = useRef(0)
  // Latest onAuthRejected without re-running the connect effect.
  const onAuthRejectedRef = useRef(onAuthRejected)
  onAuthRejectedRef.current = onAuthRejected

  const setVad = useCallback((state: VadState, p: number) => {
    setVadState(state)
    setProb(p)
  }, [])

  const idle = useCallback(() => {
    clientStateRef.current = 'IDLE'
    setCanTap(true)
  }, [])

  // Re-arm listening for the next turn WITHOUT another tap — this is what keeps a
  // conversation going hands-free once started. The server's ARMED state auto-expires
  // (arm_timeout) if you stay quiet, which drops the client back to tap-required.
  const rearm = useCallback(() => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    clientStateRef.current = 'ARMED'
    setCanTap(false)
    setStatus('listening — your turn')
    setVad('ARMED', 0)
    ws.send(JSON.stringify({ type: 'turn_start' }))
  }, [setVad])

  const enqueueAudio = useCallback(async (arrayBuffer: ArrayBuffer) => {
    if (!playCtxRef.current) {
      playCtxRef.current = new AudioContext()
      nextStartRef.current = 0
    }
    const ctx = playCtxRef.current
    const buf = await ctx.decodeAudioData(arrayBuffer.slice(0))
    if (disposedRef.current) return
    const src = ctx.createBufferSource()
    src.buffer = buf
    src.connect(ctx.destination)
    const startAt = Math.max(ctx.currentTime, nextStartRef.current)
    src.start(startAt)
    nextStartRef.current = startAt + buf.duration
  }, [])

  const handleMessage = useCallback(async (ev: MessageEvent) => {
    if (disposedRef.current) return
    if (typeof ev.data !== 'string') {
      setStatus('speaking')
      audioPromisesRef.current.push(enqueueAudio(ev.data as ArrayBuffer))
      return
    }
    const m = JSON.parse(ev.data)
    if (m.type === 'vad_state') {
      setVad(m.state, m.prob)
    } else if (m.type === 'transcript') {
      setTurns((current) => [
        ...current,
        { id: turnIdRef.current++, role: 'user', text: m.text },
        { id: turnIdRef.current++, role: 'assistant', text: '' }
      ])
      setStatus('thinking')
    } else if (m.type === 'reply') {
      setTurns((current) => {
        const next = [...current]
        const last = next[next.length - 1]
        if (last?.role === 'assistant') {
          next[next.length - 1] = { ...last, text: last.text + m.text + ' ' }
        }
        return next
      })
      setStatus('speaking')
    } else if (m.type === 'done') {
      // Wait for every queued reply chunk to finish decoding/scheduling before reading
      // nextStart, then wait out playback before listening again — otherwise the mic
      // re-arms mid-reply and picks up the bot's own trailing audio.
      await Promise.all(audioPromisesRef.current)
      audioPromisesRef.current = []
      if (disposedRef.current) return
      const ctx = playCtxRef.current
      const delayMs = ctx ? Math.max(0, (nextStartRef.current - ctx.currentTime) * 1000) : 0
      setVad('IDLE', 0)
      if (m.ending) {
        // The engine classified this reply as a goodbye — don't auto-listen for a
        // reply that was never coming.
        rearmTimerRef.current = window.setTimeout(() => {
          setStatus('conversation ended — tap to start again')
          idle()
        }, delayMs)
      } else {
        setStatus('your turn')
        rearmTimerRef.current = window.setTimeout(rearm, delayMs)
      }
    } else if (m.type === 'arm_timeout') {
      setStatus('no speech — tap again')
      setVad('IDLE', 0)
      idle()
    } else if (m.type === 'vad_error') {
      setStatus('VAD unavailable')
      idle()
    }
  }, [enqueueAudio, idle, rearm, setVad])

  const connect = useCallback(() => {
    if (disposedRef.current) return
    setStatus('connecting…')
    // Token as the Sec-WebSocket-Protocol value — never a query param (query strings
    // land verbatim in nginx/uvicorn access logs). Same transport as the tablet.
    const ws = new WebSocket(`${wsUrl}/ws-stream`, [deviceToken])
    ws.binaryType = 'arraybuffer'
    ws.onopen = () => {
      if (disposedRef.current) return
      setIsConnected(true)
      setStatus('tap to begin')
      idle()
    }
    ws.onmessage = (ev) => void handleMessage(ev)
    ws.onclose = (ev) => {
      if (disposedRef.current) return
      setIsConnected(false)
      setCanTap(false)
      micOpenRef.current = false // the server drops per-connection state; a new socket needs a new start frame
      if (ev.code === WS_CLOSE_UNAUTHORIZED) {
        setStatus('device token rejected — re-provision')
        onAuthRejectedRef.current()
      } else {
        setStatus('disconnected')
      }
    }
    wsRef.current = ws
  }, [wsUrl, deviceToken, handleMessage, idle])

  // Opt-in location, mirroring tap_index.html: fire-and-forget, a turn never waits on
  // it, and denial simply leaves the server's default location in place.
  const sendLocation = useCallback(() => {
    const ws = wsRef.current
    if (!navigator.geolocation || !ws || ws.readyState !== WebSocket.OPEN) return
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        if (disposedRef.current || ws.readyState !== WebSocket.OPEN) return
        ws.send(JSON.stringify({
          type: 'location_state',
          lat: pos.coords.latitude,
          lon: pos.coords.longitude,
          accuracy_m: Math.round(pos.coords.accuracy),
          source: 'browser'
        }))
      },
      (err) => console.info(`location unavailable (${err.message}) — server default applies`),
      { enableHighAccuracy: false, timeout: 5000, maximumAge: 600000 }
    )
  }, [])

  // Open the mic ONCE and stream continuously (never gated by the tap button). Requires
  // a user gesture, so it happens on the first tap; continuous audio is what lets the
  // server's prespeech ring recover the clipped onset of an utterance.
  const openMic = useCallback(async () => {
    if (micOpenRef.current) return
    if (!navigator.mediaDevices?.getUserMedia) {
      // Browsers expose the mic only in secure contexts: HTTPS or localhost. A LAN
      // IP/hostname over plain http gets no navigator.mediaDevices at all.
      throw new Error('microphone blocked — open the dashboard via localhost or HTTPS')
    }
    if (!capCtxRef.current) {
      const capCtx = new AudioContext({ sampleRate: 16000 })
      const micStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      })
      if (disposedRef.current) {
        micStream.getTracks().forEach((track) => track.stop())
        void capCtx.close()
        return
      }
      const source = capCtx.createMediaStreamSource(micStream)
      const workletUrl = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }))
      try {
        await capCtx.audioWorklet.addModule(workletUrl)
      } finally {
        URL.revokeObjectURL(workletUrl)
      }
      const recNode = new AudioWorkletNode(capCtx, 'recorder')
      recNode.port.onmessage = (e) => {
        const f = e.data as Float32Array
        const i16 = new Int16Array(f.length)
        for (let i = 0; i < f.length; i++) {
          const s = Math.max(-1, Math.min(1, f[i]))
          i16[i] = s * 32767
        }
        const ws = wsRef.current
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(i16.buffer)
      }
      source.connect(recNode)
      recNode.connect(capCtx.destination) // keep the graph pulling (worklet emits silence)
      capCtxRef.current = capCtx
      micStreamRef.current = micStream
    }
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    ws.send(JSON.stringify({ type: 'start', sampleRate: capCtxRef.current.sampleRate }))
    sendLocation()
    micOpenRef.current = true
  }, [sendLocation])

  const tap = useCallback(() => {
    void (async () => {
      if (clientStateRef.current !== 'IDLE') return // disabled while ARMED/SPEECH/thinking
      const ws = wsRef.current
      if (!ws || ws.readyState !== WebSocket.OPEN) return
      try {
        await openMic()
      } catch (error) {
        // Permission denied, insecure context, or no input device: say so instead of
        // swallowing the exception and looking like a dead button.
        const detail = error instanceof Error ? error.message : String(error)
        setStatus(detail.includes('microphone') ? detail : `microphone error: ${detail}`)
        return
      }
      // Barge-in is out of scope: hard-stop any current playback so the reply doesn't
      // talk over you.
      if (playCtxRef.current) {
        try {
          await playCtxRef.current.close()
        } catch {
          // already closed
        }
        playCtxRef.current = null
        nextStartRef.current = 0
      }
      rearm()
    })()
  }, [openMic, rearm])

  const reconnect = useCallback(() => {
    const ws = wsRef.current
    if (ws && ws.readyState !== WebSocket.CLOSED) return
    setTurns([])
    connect()
  }, [connect])

  useEffect(() => {
    disposedRef.current = false
    connect()
    return () => {
      // Fast Refresh remounts during development: everything opened here must close.
      disposedRef.current = true
      if (rearmTimerRef.current != null) window.clearTimeout(rearmTimerRef.current)
      const ws = wsRef.current
      if (ws) {
        ws.onmessage = null
        ws.onclose = null
        ws.close()
        wsRef.current = null
      }
      micStreamRef.current?.getTracks().forEach((track) => track.stop())
      micStreamRef.current = null
      void capCtxRef.current?.close().catch(() => undefined)
      capCtxRef.current = null
      void playCtxRef.current?.close().catch(() => undefined)
      playCtxRef.current = null
      micOpenRef.current = false
      audioPromisesRef.current = []
    }
  }, [connect])

  return { status, vadState, prob, turns, canTap, isConnected, tap, reconnect }
}
