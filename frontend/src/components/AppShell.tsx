"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { logout, type Me } from "../lib/api";
import { clearStoredSession } from "../lib/auth";

const NAV_LINKS = [
  { href: "/", label: "Chats" },
  { href: "/users", label: "Users", adminOnly: true },
  { href: "/activities", label: "Activities" },
];

/** Brand banner + primary nav + signed-in identity + logout, shared by every authed
 * page. Logout revokes the token server-side before clearing local storage. */
export default function AppShell({
  user,
  children,
}: {
  user: Me;
  children: React.ReactNode;
}) {
  const router = useRouter();
  const pathname = usePathname();

  async function handleLogout() {
    try {
      await logout();
    } catch {
      // The token may already be expired or revoked; clearing locally is what matters.
    }
    clearStoredSession();
    router.replace("/login");
  }

  return (
    <div className="app-shell">
      <main className="main-panel">
        <section className="brand-banner">
          <div className="brand-banner-row">
            <div>
              <h1>Robin Dashboard</h1>
            </div>
            <div className="brand-banner-actions">
              <nav className="page-nav" aria-label="Primary">
                {NAV_LINKS.filter((link) => !link.adminOnly || user.is_admin).map((link) => (
                  <Link
                    key={link.href}
                    className={
                      pathname === link.href ? "nav-link nav-link-active" : "nav-link"
                    }
                    href={link.href}
                  >
                    {link.label}
                  </Link>
                ))}
              </nav>
              <span className="toolbar-meta">
                Signed in as <strong>{user.display_name}</strong>
              </span>
              <button className="secondary-button" onClick={handleLogout} type="button">
                Log out
              </button>
            </div>
          </div>
        </section>

        {children}
      </main>
    </div>
  );
}
