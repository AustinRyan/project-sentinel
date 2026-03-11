"""Tests for cryptographic proof chain."""
from __future__ import annotations

import json

import pytest

from janus.core.proof import ProofChain


@pytest.fixture
def chain() -> ProofChain:
    return ProofChain()


def test_empty_chain(chain: ProofChain) -> None:
    assert chain.get_chain("s1") == []


def test_add_single_node(chain: ProofChain) -> None:
    chain.add(
        session_id="s1",
        agent_id="agent-1",
        tool_name="read_file",
        tool_input={"path": "/etc/config"},
        verdict="allow",
        risk_score=2.0,
        risk_delta=2.0,
    )
    nodes = chain.get_chain("s1")
    assert len(nodes) == 1
    assert nodes[0].step == 1
    assert nodes[0].tool_name == "read_file"
    assert nodes[0].verdict == "allow"
    assert nodes[0].parent_hash == ""


def test_chain_links_parent_hash(chain: ProofChain) -> None:
    chain.add(
        session_id="s1", agent_id="a",
        tool_name="read_file", tool_input={},
        verdict="allow", risk_score=2.0, risk_delta=2.0,
    )
    chain.add(
        session_id="s1", agent_id="a",
        tool_name="send_email", tool_input={},
        verdict="block", risk_score=37.0, risk_delta=35.0,
    )
    nodes = chain.get_chain("s1")
    assert len(nodes) == 2
    assert nodes[1].parent_hash == nodes[0].node_id
    assert nodes[0].parent_hash == ""


def test_verify_valid_chain(chain: ProofChain) -> None:
    chain.add(
        session_id="s1", agent_id="a",
        tool_name="t1", tool_input={},
        verdict="allow", risk_score=5.0, risk_delta=5.0,
    )
    chain.add(
        session_id="s1", agent_id="a",
        tool_name="t2", tool_input={"x": 1},
        verdict="block", risk_score=40.0, risk_delta=35.0,
    )
    chain.add(
        session_id="s1", agent_id="a",
        tool_name="t3", tool_input={},
        verdict="allow", risk_score=42.0, risk_delta=2.0,
    )
    assert chain.verify("s1") is True


def test_verify_empty_chain(chain: ProofChain) -> None:
    assert chain.verify("nonexistent") is True


def test_verify_detects_tampering(chain: ProofChain) -> None:
    chain.add(
        session_id="s1", agent_id="a",
        tool_name="t1", tool_input={},
        verdict="allow", risk_score=5.0, risk_delta=5.0,
    )
    chain.add(
        session_id="s1", agent_id="a",
        tool_name="t2", tool_input={},
        verdict="block", risk_score=40.0, risk_delta=35.0,
    )
    nodes = chain.get_chain("s1")
    nodes[0].verdict = "block"  # TAMPERED
    assert chain.verify("s1") is False


def test_export_returns_json(chain: ProofChain) -> None:
    chain.add(
        session_id="s1", agent_id="a",
        tool_name="t1", tool_input={},
        verdict="allow", risk_score=5.0, risk_delta=5.0,
    )
    exported = chain.export("s1")
    data = json.loads(exported)
    assert isinstance(data, list)
    assert len(data) == 1
    assert "node_id" in data[0]
    assert "parent_hash" in data[0]


def test_sessions_isolated(chain: ProofChain) -> None:
    chain.add(
        session_id="s1", agent_id="a",
        tool_name="t1", tool_input={},
        verdict="allow", risk_score=5.0, risk_delta=5.0,
    )
    chain.add(
        session_id="s2", agent_id="b",
        tool_name="t2", tool_input={},
        verdict="block", risk_score=40.0, risk_delta=40.0,
    )
    assert len(chain.get_chain("s1")) == 1
    assert len(chain.get_chain("s2")) == 1


def test_node_id_is_deterministic(chain: ProofChain) -> None:
    chain.add(
        session_id="s1", agent_id="a",
        tool_name="t1", tool_input={"k": "v"},
        verdict="allow", risk_score=5.0, risk_delta=5.0,
    )
    node = chain.get_chain("s1")[0]
    assert len(node.node_id) == 64  # SHA-256 hex
