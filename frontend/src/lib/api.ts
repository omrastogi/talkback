import { clearStoredSession, getAuthToken } from "./auth";

// Default to the same-origin /api proxy (see next.config.ts rewrites).
const rawBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "/api";

const API_BASE_URL = rawBaseUrl.replace(/\/$/, "");
const REQUEST_TIMEOUT_MS = 10000;

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function isSessionExpired(error: unknown): boolean {
  return error instanceof ApiError && error.status === 401;
}

// FastAPI puts errors in `detail`: a string, or for 422 an array of {loc, msg} objects.
function extractDetail(body: unknown, fallback: string): string {
  if (!body || typeof body !== "object") return fallback;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const msgs = detail
      .map((item) => (item && typeof item === "object" ? (item as { msg?: string }).msg : null))
      .filter((msg): msg is string => Boolean(msg));
    if (msgs.length) return msgs.join("; ");
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getAuthToken();
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(init?.headers || {}),
      },
      ...init,
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Request timed out while contacting the backend.");
    }
    throw new Error("Could not reach the backend API.");
  } finally {
    window.clearTimeout(timeoutId);
  }

  if (response.status === 401 && token) {
    clearStoredSession();
    throw new ApiError(401, "Your session expired. Please sign in again.");
  }

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      message = extractDetail(await response.json(), message);
    } catch {
      // Ignore JSON parse failures and keep the default message.
    }
    throw new ApiError(response.status, message);
  }

  return response.json() as Promise<T>;
}

export type JsonObject = Record<string, unknown>;

// ---------------------------------------------------------------------------
// Auth

export interface LoginResponse {
  token: string;
  account_id: number;
  display_name: string;
  is_admin: boolean;
}

export interface Me {
  account_id: number;
  email: string;
  display_name: string;
  is_admin: boolean;
}

export function login(payload: { email: string; password: string }) {
  return request<LoginResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function logout() {
  return request<{ revoked: boolean }>("/auth/logout", { method: "POST" });
}

export function fetchMe() {
  return request<Me>("/auth/me");
}

// ---------------------------------------------------------------------------
// Profiles

export interface Profile {
  id: number;
  display_name: string;
  timezone: string;
  voice: string;
  speech_rate: number;
  context: JsonObject;
  active: boolean;
  role: string; // caller's role: "owner" | "viewer" | "admin" (admin = unlinked admin access)
  created_at: string;
  updated_at: string;
}

export interface ProfilePatch {
  display_name?: string;
  timezone?: string;
  voice?: string;
  speech_rate?: number;
  context?: JsonObject;
}

export function fetchProfiles() {
  return request<{ profiles: Profile[] }>("/profiles");
}

export function fetchProfile(profileId: number) {
  return request<Profile>(`/profiles/${profileId}`);
}

export function patchProfile(profileId: number, patch: ProfilePatch) {
  return request<Profile>(`/profiles/${profileId}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
}

// ---------------------------------------------------------------------------
// Admin: provisioning

export interface AccountInfo {
  id: number;
  email: string;
  display_name: string;
  is_admin: boolean;
  created_at: string;
}

export interface CreatedProfile {
  id: number;
  display_name: string;
  timezone: string;
  voice: string;
  speech_rate: number;
  context: JsonObject;
  active: boolean;
  created_at: string;
  updated_at: string;
}

export interface LinkInfo {
  account_id: number;
  profile_id: number;
  role: string;
  created_at: string;
}

export interface TokenInfo {
  id: number;
  kind: string;
  account_id: number | null;
  profile_id: number | null;
  label: string | null;
  last_used_at: string | null;
  revoked_at: string | null;
  created_at: string;
}

export interface DeviceTokenIssue {
  token: string; // the raw device token; shown exactly once, never recoverable
  token_id: number;
  profile_id: number;
  label: string;
  created_at: string;
}

export function fetchAccounts() {
  return request<{ accounts: AccountInfo[] }>("/accounts");
}

export function createAccount(payload: {
  email: string;
  password: string;
  display_name: string;
  is_admin?: boolean;
}) {
  return request<AccountInfo>("/accounts", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function createProfile(payload: {
  display_name: string;
  timezone?: string;
  voice?: string;
  speech_rate?: number;
  context?: JsonObject;
}) {
  return request<CreatedProfile>("/profiles", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function fetchProfileLinks(profileId: number) {
  return request<{ links: LinkInfo[] }>(`/profiles/${profileId}/links`);
}

export function linkAccountProfile(
  profileId: number,
  payload: { account_id: number; role?: "owner" | "viewer" },
) {
  return request<LinkInfo>(`/profiles/${profileId}/links`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function unlinkAccountProfile(profileId: number, accountId: number) {
  return request<{ deleted: boolean }>(`/profiles/${profileId}/links/${accountId}`, {
    method: "DELETE",
  });
}

export function issueDeviceToken(profileId: number, label: string) {
  return request<DeviceTokenIssue>(`/profiles/${profileId}/device-tokens`, {
    method: "POST",
    body: JSON.stringify({ label }),
  });
}

export function fetchTokens(params: { profile_id?: number; include_revoked?: boolean } = {}) {
  const query = new URLSearchParams();
  if (params.profile_id != null) query.set("profile_id", String(params.profile_id));
  if (params.include_revoked) query.set("include_revoked", "true");
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return request<{ tokens: TokenInfo[] }>(`/tokens${suffix}`);
}

export function revokeToken(tokenId: number) {
  return request<{ revoked: boolean }>(`/tokens/${tokenId}/revoke`, { method: "POST" });
}

/** A short spoken sample of one voice, as a WAV blob. First request for a voice can take
 * several seconds (the server synthesizes it, downloading the voice weights if needed). */
export async function fetchVoicePreview(voiceId: string): Promise<Blob> {
  const token = getAuthToken();
  const response = await fetch(`${API_BASE_URL}/voices/${voiceId}/preview`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!response.ok) {
    let message = `Preview failed with status ${response.status}`;
    try {
      message = extractDetail(await response.json(), message);
    } catch {
      // keep default
    }
    throw new ApiError(response.status, message);
  }
  return response.blob();
}

// ---------------------------------------------------------------------------
// Chats

export interface SessionSummary {
  session_id: string;
  started_at: string;
  last_at: string;
  turn_count: number;
  sources: string[];
}

export interface Turn {
  id: number;
  session_id: string;
  turn_index: number;
  role: string; // "user" | "assistant" | "system"
  content: string;
  source: string; // "voice" | "proactive" | "tool"
  latency_ms: number | null;
  meta: JsonObject;
  created_at: string;
}

export function fetchSessions(
  profileId: number,
  params: { before?: string; limit?: number; include_diagnostic?: boolean } = {},
) {
  const query = new URLSearchParams();
  if (params.before) query.set("before", params.before);
  if (params.limit != null) query.set("limit", String(params.limit));
  if (params.include_diagnostic) query.set("include_diagnostic", "true");
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return request<{ sessions: SessionSummary[] }>(`/profiles/${profileId}/sessions${suffix}`);
}

export function fetchTurns(
  profileId: number,
  params: {
    session_id?: string;
    before?: string;
    limit?: number;
    include_diagnostic?: boolean;
  } = {},
) {
  const query = new URLSearchParams();
  if (params.session_id) query.set("session_id", params.session_id);
  if (params.before) query.set("before", params.before);
  if (params.limit != null) query.set("limit", String(params.limit));
  if (params.include_diagnostic) query.set("include_diagnostic", "true");
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return request<{ turns: Turn[] }>(`/profiles/${profileId}/turns${suffix}`);
}

// ---------------------------------------------------------------------------
// Activities

export interface ActivityDay {
  day: string;
  sessions: number;
  user_turns: number;
  assistant_turns: number;
  proactive_turns: number;
  voice_turns: number;
  avg_latency_ms: number | null;
}

export interface Activity {
  timezone: string;
  days: ActivityDay[];
  last_active_at: string | null;
}

export function fetchActivity(
  profileId: number,
  params: { date_from?: string; date_to?: string; include_diagnostic?: boolean } = {},
) {
  const query = new URLSearchParams();
  if (params.date_from) query.set("date_from", params.date_from);
  if (params.date_to) query.set("date_to", params.date_to);
  if (params.include_diagnostic) query.set("include_diagnostic", "true");
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return request<Activity>(`/profiles/${profileId}/activity${suffix}`);
}

// ---------------------------------------------------------------------------
// Wake word enrollment

export interface WakeClip {
  id: number;
  label: string; // "positive" (the wake word) | "negative" (ordinary speech)
  duration_s: number;
  sha256: string;
  created_at: string;
}

export interface WakeTraining {
  state: "running" | "done" | "failed";
  detail: string;
  started_at: string;
}

export interface WakeModelStatus {
  available: boolean;
  sha256: string | null;
  base_version: string | null;
  threshold: number | null;
  created_at: string | null;
  manifest: JsonObject | null;
  /** Current takes differ from the set the active model was trained on — Register unlocks. */
  clips_changed: boolean;
  positives: number;
  training: WakeTraining | null;
}

export function fetchWakeClips(profileId: number) {
  return request<{ clips: WakeClip[] }>(`/profiles/${profileId}/wake-clips`);
}

export function deleteWakeClip(profileId: number, clipId: number) {
  return request<{ deleted: boolean }>(`/profiles/${profileId}/wake-clips/${clipId}`, {
    method: "DELETE",
  });
}

export function fetchWakeModel(profileId: number) {
  return request<WakeModelStatus>(`/profiles/${profileId}/wake-model`);
}

/** The Register button: start a server-side training run on the current takes. */
export function trainWakeModel(profileId: number) {
  return request<{ started: boolean }>(`/profiles/${profileId}/wake-model/train`, {
    method: "POST",
  });
}

/** Upload one recorded take as a raw WAV body (16 kHz mono PCM16 — the trainer's input
 * contract, encoded client-side). Not JSON, so this bypasses request(). */
export async function uploadWakeClip(
  profileId: number,
  wav: Blob,
  label: "positive" | "negative" = "positive",
): Promise<WakeClip> {
  const token = getAuthToken();
  const response = await fetch(`${API_BASE_URL}/profiles/${profileId}/wake-clips?label=${label}`, {
    method: "POST",
    headers: {
      "Content-Type": "audio/wav",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: wav,
  });
  if (!response.ok) {
    let message = `Upload failed with status ${response.status}`;
    try {
      message = extractDetail(await response.json(), message);
    } catch {
      // keep default
    }
    throw new ApiError(response.status, message);
  }
  return response.json() as Promise<WakeClip>;
}

/** One stored take, as a playable WAV blob. */
export async function fetchWakeClipAudio(profileId: number, clipId: number): Promise<Blob> {
  const token = getAuthToken();
  const response = await fetch(`${API_BASE_URL}/profiles/${profileId}/wake-clips/${clipId}/audio`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!response.ok) {
    throw new ApiError(response.status, `Audio fetch failed with status ${response.status}`);
  }
  return response.blob();
}

// ---------------------------------------------------------------------------
// Live voice session (diagnostic page)

export interface Health {
  status: string;
  stt_loaded: boolean;
  tts_loaded: boolean;
  turn_mode: string; // "tap" | "hold"
}

export function fetchHealth() {
  return request<Health>("/health");
}

/** Base URL for the voice WebSocket: NEXT_PUBLIC_VOICE_WS_URL when set, else the API
 * base with ws(s) scheme — for the default same-origin "/api" that means the socket
 * rides the Next proxy (which does forward WS upgrades), so a single forwarded port
 * 3000 carries the whole dashboard, live audio included. */
export function voiceWsUrl(): string {
  const explicit = process.env.NEXT_PUBLIC_VOICE_WS_URL;
  if (explicit) return explicit.replace(/\/$/, "");
  if (/^https?:/.test(API_BASE_URL)) return API_BASE_URL.replace(/^http/, "ws");
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}${API_BASE_URL}`;
}
