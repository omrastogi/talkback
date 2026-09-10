"use client";

// The stored session is the raw /auth/login response. The token is an opaque dashboard
// token (server stores only its hash); localStorage is XSS-exposed, accepted for an
// internal research dashboard.
const AUTH_STORAGE_KEY = "robin-auth-session";

export interface AuthSession {
  token: string;
  account_id: number;
  display_name: string;
  is_admin: boolean;
}

function canUseStorage() {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

export function getStoredSession(): AuthSession | null {
  if (!canUseStorage()) return null;

  const rawValue = window.localStorage.getItem(AUTH_STORAGE_KEY);
  if (!rawValue) return null;

  try {
    return JSON.parse(rawValue) as AuthSession;
  } catch {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    return null;
  }
}

export function storeSession(session: AuthSession) {
  if (!canUseStorage()) return;
  window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(session));
}

export function clearStoredSession() {
  if (!canUseStorage()) return;
  window.localStorage.removeItem(AUTH_STORAGE_KEY);
}

export function getAuthToken() {
  return getStoredSession()?.token || null;
}
