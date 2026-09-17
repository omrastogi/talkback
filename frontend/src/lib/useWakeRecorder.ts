'use client'

/**
 * Fixed-length wake-word take recorder for the enrollment page. Captures at 16 kHz mono
 * through the same AudioContext + worklet graph useVoiceSession uses, and encodes PCM16
 * WAV client-side — the trainer's exact input contract (oww-train/enroll_train.py), so
 * the server stores bit-for-bit what was captured and never transcodes.
 *
 * The mic opens on the first record() (a user gesture) and stays open between takes;
 * everything closes on unmount (Fast Refresh remounts during development, and a leaked
 * AudioContext keeps the mic hot).
 */
import { useCallback, useEffect, useRef, useState } from 'react'

// Same worklet as useVoiceSession: forward each 128-sample float32 block to JS.
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

export function encodeWav(chunks: Float32Array[], sampleRate: number): Blob {
  let sampleCount = 0
  for (const chunk of chunks) sampleCount += chunk.length
  const buffer = new ArrayBuffer(44 + sampleCount * 2)
  const view = new DataView(buffer)
  const ascii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i))
  }
  ascii(0, 'RIFF'); view.setUint32(4, 36 + sampleCount * 2, true); ascii(8, 'WAVE')
  ascii(12, 'fmt '); view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true) // byte rate
  view.setUint16(32, 2, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  ascii(36, 'data'); view.setUint32(40, sampleCount * 2, true)
  let offset = 44
  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, chunk[i]))
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    }
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

export interface WakeRecorder {
  isRecording: boolean
  /** Record for `seconds` and resolve to the encoded WAV. Rejects on mic problems. */
  record: (seconds: number) => Promise<Blob>
}

export function useWakeRecorder(): WakeRecorder {
  const [isRecording, setIsRecording] = useState(false)
  const ctxRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Float32Array[] | null>(null) // non-null only while recording
  const disposedRef = useRef(false)

  const openMic = useCallback(async () => {
    if (ctxRef.current) return
    if (!navigator.mediaDevices?.getUserMedia) {
      // Browsers expose the mic only in secure contexts: HTTPS or localhost.
      throw new Error('microphone blocked — open the dashboard via localhost or HTTPS')
    }
    const ctx = new AudioContext({ sampleRate: 16000 })
    const stream = await navigator.mediaDevices.getUserMedia({
      // No aggressive processing: the tablet's far-field mic won't have it either, and
      // the trainer's augmentation handles channel simulation.
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true }
    })
    if (disposedRef.current) {
      stream.getTracks().forEach((track) => track.stop())
      void ctx.close()
      return
    }
    const source = ctx.createMediaStreamSource(stream)
    const workletUrl = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }))
    try {
      await ctx.audioWorklet.addModule(workletUrl)
    } finally {
      URL.revokeObjectURL(workletUrl)
    }
    const recNode = new AudioWorkletNode(ctx, 'recorder')
    recNode.port.onmessage = (e) => {
      chunksRef.current?.push(e.data as Float32Array)
    }
    source.connect(recNode)
    recNode.connect(ctx.destination) // keep the graph pulling (worklet emits silence)
    ctxRef.current = ctx
    streamRef.current = stream
  }, [])

  const record = useCallback(async (seconds: number): Promise<Blob> => {
    if (chunksRef.current) throw new Error('already recording')
    await openMic()
    const ctx = ctxRef.current
    if (!ctx) throw new Error('recorder disposed')
    // Autoplay policy can leave a context suspended; recording from one yields silence.
    if (ctx.state === 'suspended') await ctx.resume()
    chunksRef.current = []
    setIsRecording(true)
    try {
      await new Promise((resolve) => setTimeout(resolve, seconds * 1000))
      return encodeWav(chunksRef.current ?? [], ctx.sampleRate)
    } finally {
      chunksRef.current = null
      setIsRecording(false)
    }
  }, [openMic])

  useEffect(() => {
    disposedRef.current = false
    return () => {
      disposedRef.current = true
      chunksRef.current = null
      streamRef.current?.getTracks().forEach((track) => track.stop())
      streamRef.current = null
      void ctxRef.current?.close().catch(() => undefined)
      ctxRef.current = null
    }
  }, [])

  return { isRecording, record }
}
