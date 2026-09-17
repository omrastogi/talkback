import type { NextConfig } from "next";

function normalizeBasePath(value?: string) {
  if (!value) return undefined;

  const trimmed = value.trim();
  if (!trimmed || trimmed === "/") return undefined;

  const withLeadingSlash = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return withLeadingSlash.replace(/\/+$/, "");
}

const nextConfig: NextConfig = {
  reactStrictMode: true,
  basePath: normalizeBasePath(process.env.NEXT_BASE_PATH),

  // Proxy API routes to the FastAPI server so the browser only ever talks to this
  // origin — works through a single forwarded port (VS Code / SSH tunnels) and makes
  // CORS irrelevant. Used when NEXT_PUBLIC_API_BASE_URL is empty/unset.
  // Everything lives under /api so backend routes can never collide with page routes
  // (the bare prefixes did: the /accounts page shadowed the /accounts API).
  async rewrites() {
    const target = (process.env.API_PROXY_TARGET || "http://localhost:8000").replace(/\/$/, "");
    return [{ source: "/api/:path*", destination: `${target}/:path*` }];
  },
};

export default nextConfig;
