"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { fetchMe, login } from "../../lib/api";
import { clearStoredSession, getStoredSession, storeSession } from "../../lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCheckingSession, setIsCheckingSession] = useState(true);

  useEffect(() => {
    async function hydrateExistingSession() {
      const session = getStoredSession();
      if (!session?.token) {
        setIsCheckingSession(false);
        return;
      }

      try {
        await fetchMe();
        router.replace("/");
      } catch {
        clearStoredSession();
        setIsCheckingSession(false);
      }
    }

    void hydrateExistingSession();
  }, [router]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!email.trim() || !password || isSubmitting) return;

    setIsSubmitting(true);
    setErrorMessage("");

    try {
      const response = await login({ email: email.trim(), password });
      storeSession(response);
      router.replace("/");
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Unable to sign in.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  if (isCheckingSession) {
    return (
      <div className="auth-shell">
        <div className="auth-card auth-card-compact">
          <p className="eyebrow">Authentication</p>
          <h1>Checking session</h1>
          <p className="auth-copy">Loading your saved access token.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-shell">
      <main className="auth-card">
        <div className="auth-intro">
          <p className="eyebrow">Robin</p>
          <h1>Sign in to the dashboard</h1>
          <p className="auth-copy">
            Use an account provisioned by the research team. Public signups are not enabled.
          </p>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label className="field">
            <span>Email</span>
            <input
              autoComplete="email"
              name="email"
              type="email"
              onChange={(event) => setEmail(event.target.value)}
              placeholder="Enter your email"
              value={email}
            />
          </label>

          <label className="field">
            <span>Password</span>
            <input
              autoComplete="current-password"
              name="password"
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Enter your password"
              type="password"
              value={password}
            />
          </label>

          {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}

          <button className="send-button auth-submit" disabled={isSubmitting} type="submit">
            {isSubmitting ? "Signing in..." : "Sign in"}
          </button>
        </form>
      </main>
    </div>
  );
}
