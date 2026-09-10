"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { fetchMe, type Me } from "./api";
import { clearStoredSession, getStoredSession } from "./auth";

/** Client-side auth guard shared by every page: validates the stored token against
 * /auth/me on mount, redirecting to /login when absent or rejected. With requireAdmin,
 * non-admins are sent to / (the backend enforces 403 regardless). */
export function useAuthGuard(options: { requireAdmin?: boolean } = {}) {
  const { requireAdmin = false } = options;
  const router = useRouter();
  const [user, setUser] = useState<Me | null>(null);
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function initializeAuth() {
      if (!getStoredSession()?.token) {
        router.replace("/login");
        return;
      }

      try {
        const me = await fetchMe();
        if (cancelled) return;
        if (requireAdmin && !me.is_admin) {
          router.replace("/");
          return;
        }
        setUser(me);
      } catch {
        if (cancelled) return;
        clearStoredSession();
        router.replace("/login");
      } finally {
        if (!cancelled) setIsChecking(false);
      }
    }

    void initializeAuth();
    return () => {
      cancelled = true;
    };
  }, [router, requireAdmin]);

  return { user, isChecking };
}
