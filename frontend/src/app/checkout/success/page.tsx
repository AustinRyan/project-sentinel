"use client";

import { useState, useEffect, useCallback } from "react";
import { Check, Copy, ArrowRight, AlertCircle } from "lucide-react";
import { Instrument_Serif, Outfit, Fira_Code } from "next/font/google";

const serif = Instrument_Serif({
  subsets: ["latin"],
  weight: "400",
  style: ["normal", "italic"],
  variable: "--font-serif",
  display: "swap",
});

const sans = Outfit({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const mono = Fira_Code({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

interface SessionData {
  license_key: string;
  tier: string;
  customer_email: string;
  trial_ends_at: string;
}

export default function CheckoutSuccess() {
  const [session, setSession] = useState<SessionData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sessionId = params.get("session_id");

    if (!sessionId) {
      setError(true);
      setLoading(false);
      return;
    }

    // Retry a few times — the webhook may arrive after the redirect
    let attempts = 0;
    const maxAttempts = 8;
    const delayMs = 2000;

    const poll = () => {
      attempts++;
      fetch(`/api/billing/session/${sessionId}`)
        .then((res) => {
          if (!res.ok) throw new Error("Not found");
          return res.json();
        })
        .then((data) => {
          setSession(data);
          setLoading(false);
        })
        .catch(() => {
          if (attempts < maxAttempts) {
            setTimeout(poll, delayMs);
          } else {
            setError(true);
            setLoading(false);
          }
        });
    };

    // Start after a brief delay to give the webhook time to arrive
    setTimeout(poll, 1500);
  }, []);

  const handleCopy = useCallback(() => {
    if (!session) return;
    navigator.clipboard.writeText(session.license_key);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [session]);

  const trialEnd = session?.trial_ends_at
    ? new Date(session.trial_ends_at).toLocaleDateString("en-US", {
        month: "long",
        day: "numeric",
        year: "numeric",
      })
    : null;

  return (
    <div
      className={`${serif.variable} ${sans.variable} ${mono.variable} flex min-h-screen items-center justify-center px-4`}
      style={{
        fontFamily: "var(--font-sans)",
        backgroundColor: "#06080e",
        color: "#e4e7ef",
      }}
    >
      <div className="w-full max-w-lg">
        {loading && (
          <div className="flex flex-col items-center gap-4">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-[#5eead4]/20 border-t-[#5eead4]" />
            <p className="text-sm text-[#636880]">Loading your license...</p>
          </div>
        )}

        {error && (
          <div className="rounded-2xl border border-[#1e2233] bg-[#0a0c13] p-8 text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-[#f59e0b]/10">
              <AlertCircle size={24} className="text-[#f59e0b]" />
            </div>
            <h1
              className="text-2xl font-bold"
              style={{ fontFamily: "var(--font-serif)" }}
            >
              Check Your Email
            </h1>
            <p className="mt-3 text-sm text-[#636880] leading-relaxed">
              We couldn&apos;t load your session, but don&apos;t worry &mdash;
              your license key has been sent to your email. Check your inbox
              (and spam folder) for a message from Janus Security.
            </p>
            <a
              href="/landing"
              className="mt-6 inline-flex items-center gap-2 text-sm text-[#5eead4] hover:underline"
            >
              Back to home
              <ArrowRight size={14} />
            </a>
          </div>
        )}

        {session && (
          <div className="rounded-2xl border border-[#1e2233] bg-[#0a0c13] p-8">
            {/* Success header */}
            <div className="mb-6 text-center">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-[#5eead4]/10">
                <Check size={24} className="text-[#5eead4]" />
              </div>
              <h1
                className="text-2xl font-bold"
                style={{ fontFamily: "var(--font-serif)" }}
              >
                Welcome to Janus Pro
              </h1>
              <p className="mt-2 text-sm text-[#636880]">
                Your 14-day free trial is active
                {trialEnd && <> until {trialEnd}</>}.
              </p>
            </div>

            {/* License key */}
            <div className="overflow-hidden rounded-xl border border-[#1e2233] bg-[#06080e] p-5">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-[11px] font-bold uppercase tracking-wider text-[#636880]">
                  License Key
                </span>
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1.5 rounded-md bg-[#1e2233] px-2.5 py-1 text-[11px] text-[#94a3b8] transition-colors hover:bg-[#2a2f45] hover:text-white"
                >
                  {copied ? (
                    <>
                      <Check size={12} className="text-[#5eead4]" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy size={12} />
                      Copy
                    </>
                  )}
                </button>
              </div>
              <code
                className="block overflow-wrap-anywhere break-all text-[13px] leading-relaxed text-[#5eead4]"
                style={{ fontFamily: "var(--font-mono)", overflowWrap: "anywhere", wordBreak: "break-all" }}
              >
                {session.license_key}
              </code>
            </div>

            {/* Activation instructions */}
            <div className="mt-6 space-y-4">
              <h2 className="text-sm font-bold text-[#94a3b8]">
                Activate your license
              </h2>

              <div className="rounded-lg border border-[#1e2233] bg-[#06080e] p-4">
                <p className="mb-2 text-xs font-bold text-[#636880]">
                  Option 1 &mdash; Config file
                </p>
                <code
                  className="block text-xs leading-relaxed text-[#a78bfa]"
                  style={{ fontFamily: "var(--font-mono)" }}
                >
                  <span className="text-[#636880]"># janus.toml</span>
                  <br />
                  [license]
                  <br />
                  key = &quot;{session.license_key}&quot;
                </code>
              </div>

              <div className="rounded-lg border border-[#1e2233] bg-[#06080e] p-4">
                <p className="mb-2 text-xs font-bold text-[#636880]">
                  Option 2 &mdash; API call
                </p>
                <code
                  className="block text-xs leading-relaxed text-[#a78bfa]"
                  style={{ fontFamily: "var(--font-mono)" }}
                >
                  POST /api/license/activate
                  <br />
                  {`{"license_key": "${session.license_key}"}`}
                </code>
              </div>
            </div>

            {/* Footer */}
            <div className="mt-6 flex items-center justify-between border-t border-[#1e2233] pt-5">
              <p className="text-xs text-[#475569]">
                Sent to {session.customer_email}
              </p>
              <a
                href="/landing"
                className="flex items-center gap-1.5 text-xs text-[#5eead4] hover:underline"
              >
                Back to home
                <ArrowRight size={12} />
              </a>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
