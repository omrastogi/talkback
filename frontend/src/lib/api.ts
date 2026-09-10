import { clearStoredSession, getAuthToken } from "./auth";

const rawBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "";

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
  params: { before?: string; limit?: number } = {},
) {
  const query = new URLSearchParams();
  if (params.before) query.set("before", params.before);
  if (params.limit != null) query.set("limit", String(params.limit));
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return request<{ sessions: SessionSummary[] }>(`/profiles/${profileId}/sessions${suffix}`);
}

export function fetchTurns(
  profileId: number,
  params: { session_id?: string; before?: string; limit?: number } = {},
) {
  const query = new URLSearchParams();
  if (params.session_id) query.set("session_id", params.session_id);
  if (params.before) query.set("before", params.before);
  if (params.limit != null) query.set("limit", String(params.limit));
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
  params: { date_from?: string; date_to?: string } = {},
) {
  const query = new URLSearchParams();
  if (params.date_from) query.set("date_from", params.date_from);
  if (params.date_to) query.set("date_to", params.date_to);
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return request<Activity>(`/profiles/${profileId}/activity${suffix}`);
}
