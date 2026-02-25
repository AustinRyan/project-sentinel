import { ImageResponse } from "next/og";

export const runtime = "edge";

export const alt = "Janus Security — Every tool call. Intercepted.";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          width: "100%",
          height: "100%",
          backgroundColor: "#06080e",
          padding: "60px 80px",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        {/* Background gradient */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background:
              "radial-gradient(ellipse at 50% 40%, rgba(94,234,212,0.08) 0%, transparent 60%)",
          }}
        />

        {/* Top bar */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: "4px",
            background: "linear-gradient(90deg, #5eead4, #a78bfa, #60a5fa)",
          }}
        />

        {/* Shield icon + brand */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "16px",
            marginBottom: "32px",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: "48px",
              height: "48px",
              borderRadius: "12px",
              backgroundColor: "rgba(94,234,212,0.1)",
              border: "1px solid rgba(94,234,212,0.2)",
            }}
          >
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#5eead4"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
          </div>
          <span style={{ fontSize: "28px", fontWeight: 700, color: "#e4e7ef" }}>
            Janus Security
          </span>
        </div>

        {/* Headline */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "8px",
          }}
        >
          <span
            style={{
              fontSize: "56px",
              fontWeight: 400,
              color: "#e4e7ef",
              lineHeight: 1.1,
              textAlign: "center",
            }}
          >
            Every tool call.
          </span>
          <span
            style={{
              fontSize: "56px",
              fontWeight: 400,
              fontStyle: "italic",
              color: "#5eead4",
              lineHeight: 1.1,
            }}
          >
            Intercepted.
          </span>
        </div>

        {/* Sub */}
        <span
          style={{
            marginTop: "24px",
            fontSize: "20px",
            color: "#636880",
            textAlign: "center",
          }}
        >
          10 security checks. Real-time. Every call. Zero trust.
        </span>

        {/* Stats bar */}
        <div
          style={{
            display: "flex",
            gap: "48px",
            marginTop: "40px",
            padding: "16px 32px",
            borderRadius: "16px",
            border: "1px solid rgba(30,34,51,0.8)",
            backgroundColor: "rgba(10,12,19,0.8)",
          }}
        >
          {[
            { value: "10", label: "Security checks" },
            { value: "<5ms", label: "P99 latency" },
            { value: "456", label: "Tests passing" },
          ].map((stat) => (
            <div
              key={stat.label}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "4px",
              }}
            >
              <span style={{ fontSize: "24px", fontWeight: 700, color: "#5eead4" }}>
                {stat.value}
              </span>
              <span style={{ fontSize: "13px", color: "#4a4e63" }}>{stat.label}</span>
            </div>
          ))}
        </div>
      </div>
    ),
    { ...size }
  );
}
