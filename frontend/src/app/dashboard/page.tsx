"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { apiFetch } from "@/lib/api";
import AgentSelector from "@/components/AgentSelector";
import ChatPanel from "@/components/ChatPanel";
import PipelineDetail from "@/components/PipelineDetail";
import ProofChainPanel from "@/components/ProofChainPanel";
import SecurityDashboard from "@/components/SecurityDashboard";
import SessionSidebar from "@/components/SessionSidebar";
import TaintFlowPanel from "@/components/TaintFlowPanel";
import ThreatIntelPanel from "@/components/ThreatIntelPanel";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Message {
  role: "user" | "assistant";
  content: string;
  toolCalls?: { tool_name: string; verdict: string; risk_score: number; risk_delta: number }[];
}

interface SecurityEvent {
  event_type: string;
  session_id: string;
  data: Record<string, unknown>;
  timestamp: string;
}

interface AgentInfo {
  agent_id: string;
  name: string;
  role: string;
  permissions: string[];
  is_locked: boolean;
}

interface Session {
  sessionId: string;
  agentId: string;
  agentName: string;
  agentRole: string;
  agentPermissions: string[];
  messages: Message[];
  events: SecurityEvent[];
  riskScore: number;
  ws: WebSocket | null;
}

interface ServerSession {
  session_id: string;
  agent_id: string;
  original_goal: string;
  risk_score: number;
}

interface ServerMessage {
  role: "user" | "assistant";
  content: string;
  tool_calls: { tool_name: string; verdict: string; risk_score: number; risk_delta: number }[];
}

export default function Home() {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [sessions, setSessions] = useState<Record<string, Session>>({});
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showAgentSelector, setShowAgentSelector] = useState(true);
  const [activeTab, setActiveTab] = useState<"pipeline" | "proof" | "threat">("pipeline");
  const [initializing, setInitializing] = useState(true);
  const sessionsRef = useRef(sessions);
  sessionsRef.current = sessions;

  const connectWebSocket = useCallback((sessionId: string) => {
    const wsUrl = API_BASE.replace("http", "ws");
    const ws = new WebSocket(`${wsUrl}/api/ws/session/${sessionId}`);

    ws.onmessage = (event) => {
      const secEvent: SecurityEvent = JSON.parse(event.data);
      setSessions((prev) => {
        const session = prev[sessionId];
        if (!session) return prev;
        return {
          ...prev,
          [sessionId]: {
            ...session,
            events: [...session.events, secEvent],
            riskScore:
              secEvent.data.risk_score !== undefined
                ? (secEvent.data.risk_score as number)
                : session.riskScore,
          },
        };
      });
    };

    ws.onerror = (err) => console.error("WebSocket error:", err);

    return ws;
  }, []);

  // Fetch agents and restore existing sessions on mount
  useEffect(() => {
    const initialize = async () => {
      try {
        // Fetch agents first
        const agentsResp = await apiFetch(`${API_BASE}/api/agents`);
        const agentsData: AgentInfo[] = await agentsResp.json();
        setAgents(agentsData);

        // Fetch existing sessions from the backend
        const sessionsResp = await apiFetch(`${API_BASE}/api/sessions`);
        const serverSessions: ServerSession[] = await sessionsResp.json();

        if (serverSessions.length > 0) {
          const restoredSessions: Record<string, Session> = {};

          for (const ss of serverSessions) {
            // Look up agent info
            const agent = agentsData.find((a) => a.agent_id === ss.agent_id);

            // Fetch messages for this session
            let messages: Message[] = [];
            try {
              const msgResp = await apiFetch(
                `${API_BASE}/api/sessions/${ss.session_id}/messages`
              );
              const serverMessages: ServerMessage[] = await msgResp.json();
              messages = serverMessages.map((m) => ({
                role: m.role,
                content: m.content,
                toolCalls: m.tool_calls?.length > 0 ? m.tool_calls : undefined,
              }));
            } catch {
              // Session has no messages yet — that's fine
            }

            // Connect WebSocket for real-time events
            const ws = connectWebSocket(ss.session_id);

            restoredSessions[ss.session_id] = {
              sessionId: ss.session_id,
              agentId: ss.agent_id,
              agentName: agent?.name || ss.agent_id,
              agentRole: agent?.role || "unknown",
              agentPermissions: agent?.permissions || [],
              messages,
              events: [],
              riskScore: ss.risk_score,
              ws,
            };
          }

          setSessions(restoredSessions);
          // Activate the most recent session
          const lastSession = serverSessions[serverSessions.length - 1];
          setActiveSessionId(lastSession.session_id);
          setShowAgentSelector(false);
        }
      } catch (err) {
        console.error("Failed to initialize:", err);
      } finally {
        setInitializing(false);
      }
    };
    initialize();
  }, [connectWebSocket]);

  const handleSelectAgent = useCallback(
    async (agentId: string) => {
      const agent = agents.find((a) => a.agent_id === agentId);
      if (!agent) return;

      try {
        const resp = await apiFetch(`${API_BASE}/api/sessions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ agent_id: agentId, original_goal: "" }),
        });
        const data = await resp.json();
        const sessionId: string = data.session_id;

        const ws = connectWebSocket(sessionId);

        const newSession: Session = {
          sessionId,
          agentId,
          agentName: agent.name,
          agentRole: agent.role,
          agentPermissions: agent.permissions,
          messages: [],
          events: [],
          riskScore: 0,
          ws,
        };

        setSessions((prev) => ({ ...prev, [sessionId]: newSession }));
        setActiveSessionId(sessionId);
        setShowAgentSelector(false);
      } catch (err) {
        console.error("Failed to create session:", err);
      }
    },
    [agents, connectWebSocket]
  );

  const handleCloseSession = useCallback(
    (sessionId: string) => {
      setSessions((prev) => {
        const session = prev[sessionId];
        if (session?.ws) {
          session.ws.close();
        }
        const next = { ...prev };
        delete next[sessionId];
        return next;
      });

      if (activeSessionId === sessionId) {
        const remaining = Object.keys(sessionsRef.current).filter((id) => id !== sessionId);
        if (remaining.length > 0) {
          setActiveSessionId(remaining[0]);
        } else {
          setActiveSessionId(null);
          setShowAgentSelector(true);
        }
      }
    },
    [activeSessionId]
  );

  const handleSendMessage = useCallback(
    async (message: string) => {
      if (!activeSessionId) return;

      setSessions((prev) => {
        const session = prev[activeSessionId];
        if (!session) return prev;
        return {
          ...prev,
          [activeSessionId]: {
            ...session,
            messages: [...session.messages, { role: "user", content: message }],
          },
        };
      });
      setIsLoading(true);

      try {
        const resp = await apiFetch(`${API_BASE}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: activeSessionId, message }),
        });
        const data = await resp.json();

        setSessions((prev) => {
          const session = prev[activeSessionId];
          if (!session) return prev;
          return {
            ...prev,
            [activeSessionId]: {
              ...session,
              messages: [
                ...session.messages,
                { role: "assistant", content: data.message, toolCalls: data.tool_calls },
              ],
            },
          };
        });
      } catch {
        setSessions((prev) => {
          const session = prev[activeSessionId];
          if (!session) return prev;
          return {
            ...prev,
            [activeSessionId]: {
              ...session,
              messages: [
                ...session.messages,
                { role: "assistant", content: "Error: Failed to get response." },
              ],
            },
          };
        });
      } finally {
        setIsLoading(false);
      }
    },
    [activeSessionId]
  );

  const activeSession = activeSessionId ? sessions[activeSessionId] : null;
  const sessionList = Object.values(sessions);

  // Compute global risk as max of all session risks
  const globalRisk = sessionList.length > 0
    ? Math.max(...sessionList.map((s) => s.riskScore))
    : 0;
  const totalEvents = sessionList.reduce((sum, s) => sum + s.events.length, 0);

  // Show loading state while initializing
  if (initializing) {
    return (
      <main className="h-screen flex flex-col">
        <header className="flex items-center justify-between px-6 py-3 border-b border-[#2a2a3e] bg-[#0a0a0f]">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-[#00ff88] animate-pulse" />
            <h1 className="text-lg font-bold text-[#e0e0e8]">Janus</h1>
            <span className="text-xs text-[#555570]">Autonomous Security Layer</span>
          </div>
        </header>
        <div className="flex-1 flex items-center justify-center bg-[#0a0a0f]">
          <p className="text-sm text-[#555570]">Loading sessions...</p>
        </div>
      </main>
    );
  }

  if (showAgentSelector && !activeSession) {
    return (
      <main className="h-screen flex flex-col">
        <header className="flex items-center justify-between px-6 py-3 border-b border-[#2a2a3e] bg-[#0a0a0f]">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-[#00ff88] animate-pulse" />
            <h1 className="text-lg font-bold text-[#e0e0e8]">Janus</h1>
            <span className="text-xs text-[#555570]">Autonomous Security Layer</span>
          </div>
        </header>
        <AgentSelector agents={agents} onSelectAgent={handleSelectAgent} />
      </main>
    );
  }

  return (
    <main className="h-screen flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-[#2a2a3e] bg-[#0a0a0f]">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-[#00ff88] animate-pulse" />
          <h1 className="text-lg font-bold text-[#e0e0e8]">Janus</h1>
          <span className="text-xs text-[#555570]">Autonomous Security Layer</span>
        </div>
        <div className="flex items-center gap-4 text-xs text-[#8888a0]">
          <span>
            Risk:{" "}
            <span
              style={{
                color:
                  globalRisk >= 80 ? "#ff4444" : globalRisk >= 40 ? "#ffaa00" : "#00ff88",
              }}
            >
              {globalRisk.toFixed(1)}
            </span>
          </span>
          <span>Events: {totalEvents}</span>
          <span>
            Threats:{" "}
            <span className="text-amber-400">
              {sessionList.reduce(
                (sum, s) =>
                  sum +
                  s.events.filter((e) =>
                    (e.data as Record<string, unknown>)?.reasons
                      ? ((e.data as Record<string, unknown>).reasons as string[]).some(
                          (r: string) => r.toLowerCase().includes("threat")
                        )
                      : false
                  ).length,
                0
              )}
            </span>
          </span>
          <span>Sessions: {sessionList.length}</span>
        </div>
      </header>

      {/* Main layout with sidebar */}
      <div className="flex-1 flex min-h-0">
        <SessionSidebar
          sessions={sessionList.map((s) => ({
            sessionId: s.sessionId,
            agentId: s.agentId,
            agentName: s.agentName,
            agentRole: s.agentRole,
            riskScore: s.riskScore,
          }))}
          activeSessionId={activeSessionId}
          onSelectSession={setActiveSessionId}
          onCloseSession={handleCloseSession}
          onNewSession={() => setShowAgentSelector(true)}
        />

        {activeSession ? (
          <div className="flex-1 grid grid-cols-4 min-h-0">
            <ChatPanel
              sessionId={activeSession.sessionId}
              onSendMessage={handleSendMessage}
              messages={activeSession.messages}
              isLoading={isLoading}
              agentName={activeSession.agentName}
              agentRole={activeSession.agentRole}
              agentPermissions={activeSession.agentPermissions}
            />
            <SecurityDashboard
              sessionId={activeSession.sessionId}
              events={activeSession.events}
              riskScore={activeSession.riskScore}
              agentRole={activeSession.agentRole}
              agentName={activeSession.agentName}
            />
            <TaintFlowPanel
              events={activeSession.events}
              sessionId={activeSession.sessionId}
            />
            <div className="flex flex-col h-full min-h-0 bg-[#12121a]">
              {/* Tabs */}
              <div className="flex border-b border-[#2a2a3e]">
                {(["pipeline", "proof", "threat"] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`flex-1 px-3 py-2 text-[10px] font-semibold uppercase tracking-wider transition-colors ${
                      activeTab === tab
                        ? "text-[#00ff88] border-b-2 border-[#00ff88]"
                        : "text-[#555570] hover:text-[#8888a0]"
                    }`}
                  >
                    {tab === "pipeline" ? "Pipeline" : tab === "proof" ? "Proof Chain" : "Threat Intel"}
                  </button>
                ))}
              </div>
              {/* Tab content */}
              <div className="flex-1 min-h-0">
                {activeTab === "pipeline" && (
                  <PipelineDetail events={activeSession.events} sessionId={activeSession.sessionId} />
                )}
                {activeTab === "proof" && (
                  <ProofChainPanel sessionId={activeSession.sessionId} eventCount={activeSession.events.length} />
                )}
                {activeTab === "threat" && (
                  <ThreatIntelPanel sessionId={activeSession.sessionId} eventCount={activeSession.events.length} />
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center bg-[#0a0a0f]">
            <p className="text-sm text-[#555570]">Select a session or create a new one</p>
          </div>
        )}
      </div>
    </main>
  );
}
