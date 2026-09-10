import type { Metadata } from "next";
import "../assets/main.scss";

export const metadata: Metadata = {
  title: "Robin Dashboard",
  description: "Provisioning, conversation history, and activity for Robin.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      {/* suppressHydrationWarning: browser extensions (e.g. Grammarly) inject attributes
          into <body> before hydration; only this element's attributes are exempted. */}
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
