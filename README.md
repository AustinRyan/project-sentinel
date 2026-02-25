# Janus

**Autonomous security layer for AI agents. Every tool call passes through Janus before it reaches the real tool.**

Janus sits between your AI agent and its tools, intercepting every action to enforce permissions, detect prompt injection, flag data exfiltration, and prevent privilege escalation — in real time.

```
Your AI Agent
    |  calls tool
Janus Guardian
    |  ALLOW / BLOCK / CHALLENGE
Real Tool (filesystem, API, database, etc.)
```

## Why Janus

AI agents are getting access to real tools — file systems, databases, APIs, email, code execution. One prompt injection or jailbreak and your agent is exfiltrating data, sending unauthorized emails, or running arbitrary code.

Janus stops that. Without changing your agent code.

## Quick Start

```bash
pip install janus-security
```

### As a Python SDK

```python
from janus import Guardian, JanusConfig

config = JanusConfig.from_toml("janus.toml")
guardian = await Guardian.from_config(config, registry, session_store)

# Wrap every tool call
verdict = await guardian.wrap_tool_call(
    agent_id="my-agent",
    session_id="session-123",
    original_goal="Analyze sales data",
    tool_name="send_email",
    tool_input={"to": "external@example.com", "body": "..."},
)

if verdict.verdict == "allow":
    result = execute_tool(tool_name, tool_input)
elif verdict.verdict == "block":
    print(f"Blocked: {verdict.reasons}")
```

### As an MCP Proxy

Drop Janus between Claude Desktop (or any MCP client) and your MCP tool servers:

```bash
janus-proxy config.toml
```

```toml
# config.toml
[agent]
agent_id = "claude-desktop"
role = "code"
permissions = ["read_*", "write_*", "search_*"]

[[upstream_servers]]
name = "filesystem"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/projects"]
```

Then in Claude Desktop config:
```json
{
  "mcpServers": {
    "janus": {
      "command": "janus-proxy",
      "args": ["config.toml"]
    }
  }
}
```

Every tool call now goes through the Guardian security pipeline before reaching the real tool.

### Dashboard

Janus includes a real-time security dashboard for monitoring all protected agents:

```bash
janus serve
# → http://localhost:8000
```

The dashboard shows:
- Live security events across all agents and sessions
- Risk scores, verdicts, and trend charts
- Agent status (active, locked, challenged)
- Drill-down into individual tool calls and pipeline decisions
- Audit trail with filtering and export

Self-hosted by default — your security data stays on your infrastructure.

### Configuration

```bash
janus init  # creates janus.toml
```

```toml
# janus.toml
[risk]
lock_threshold = 80.0
sandbox_threshold = 50.0

[permissions.default]
allow = ["read_*", "search_*"]
deny = ["execute_*", "write_*"]

[policy]
llm_risk_weight = 0.4

[policy.keyword_amplifiers]
"custom_danger_tool" = 35.0

[exporters.notifications.slack]
webhook_url = "https://hooks.slack.com/services/..."
min_verdict = "block"
```

## Security Pipeline

Janus runs a priority-ordered chain of security checks on every tool call:

| Check | Priority | Tier | What it does |
|-------|----------|------|-------------|
| Prompt Injection | 5 | Free | 15+ regex patterns detect injection attempts |
| Identity | 10 | Free | Verifies agent is registered and not locked |
| Permission Scope | 20 | Free | Glob-based tool permission enforcement |
| Deterministic Risk | 25 | Free | Rule-based risk scoring + pattern matching |
| Taint Tracking | 35 | Pro | Tracks sensitive data flow, blocks exfiltration |
| Predictive Risk | 38 | Pro | Matches tool sequences against known attack trajectories |
| LLM Classifier | 30 | Pro | Claude Haiku contextual risk assessment |
| Drift Detection | 40 | Pro | Detects semantic drift from original goal |
| Threat Intel | 55 | Free | Matches against known attack patterns |
| ITDR | 60 | Free | Anomaly detection, collusion, escalation tracking |

## Verdicts

Every tool call gets one of five verdicts:

- **ALLOW** - Safe to execute
- **BLOCK** - Denied, too risky
- **CHALLENGE** - Requires identity verification
- **SANDBOX** - Execute in isolated environment
- **PAUSE** - Human review required

## Features

### Free (Open Source)

- Rule-based security pipeline (permissions, risk scoring, pattern matching)
- Prompt injection detection (regex patterns)
- Agent identity and role management
- Circuit breaker (fail-safe on Guardian errors)
- MCP proxy for Claude Desktop / Cursor / any MCP client
- 6 built-in threat intelligence patterns
- Cryptographic proof chain (tamper-evident audit trail)
- Configurable policies via TOML
- Audit log export (CSV, JSON, JSONL)
- CLI tools

### Pro (Janus Cloud)

Everything in Free, plus:

- LLM-powered risk classification
- Semantic drift detection
- Causal data-flow taint tracking
- Predictive risk with lookahead
- Multi-model support (Anthropic, OpenAI, Ollama)
- Real-time notifications (Slack, email, Telegram)
- Cloud dashboard with real-time monitoring
- Compliance reports and audit exports
- SSO/SAML
- SLA + priority support

[Get started with Janus Pro](https://janus-security.dev/pricing)

## Architecture

```
janus/
  core/         # Guardian, security pipeline, verdicts
  identity/     # Agent roles, permissions, registry
  risk/         # Risk scoring engine, configurable thresholds
  drift/        # Semantic drift detection (Pro)
  circuit/      # Circuit breaker pattern
  llm/          # LLM-based security checks (Pro)
  mcp/          # MCP proxy server
  web/          # FastAPI dashboard backend
  storage/      # Session store, SQLite persistence
  forensics/    # Audit trail, trace export
  integrations/ # LangChain, OpenAI, CrewAI, MCP adapters
```

## Integrations

Janus works with any agent framework:

- **MCP** - Drop-in proxy for Claude Desktop, Cursor, Cline
- **LangChain** - Tool wrapper for LangChain agents
- **OpenAI** - Function calling proxy
- **CrewAI** - Tool integration
- **Any Python agent** - Direct SDK integration via `Guardian.wrap_tool_call()`

## Development

```bash
git clone https://github.com/AustinRyan/project-sentinel.git
cd project-sentinel
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,integrations]"
pytest tests/ -v
```

469 tests. All passing.

## License

Business Source License 1.1 (BSL-1.1). Free to use, modify, and self-host. Cannot be offered as a competing hosted service. Converts to Apache 2.0 after 4 years. See [LICENSE](LICENSE).
