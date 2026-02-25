import type { Metadata } from "next";
import "./globals.css";

const siteUrl = process.env.NEXT_PUBLIC_SITE_URL || "https://janus-security.dev";

export const metadata: Metadata = {
  title: "Janus — Autonomous Security Layer for AI Agents",
  description:
    "Every tool call. Intercepted. 10 security checks in under 5ms. Protect your AI agents from prompt injection, data exfiltration, and privilege escalation.",
  metadataBase: new URL(siteUrl),
  openGraph: {
    type: "website",
    title: "Janus — Autonomous Security Layer for AI Agents",
    description:
      "10 security checks. Real-time. Every call. Zero trust. Protect your AI agents from prompt injection, data exfiltration, and privilege escalation.",
    siteName: "Janus Security",
    url: siteUrl,
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Janus Security — Every tool call. Intercepted.",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Janus — Autonomous Security Layer for AI Agents",
    description:
      "10 security checks. Real-time. Every call. Zero trust. Protect your AI agents from prompt injection, data exfiltration, and privilege escalation.",
    images: ["/og-image.png"],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#0a0a0f]">{children}</body>
    </html>
  );
}
