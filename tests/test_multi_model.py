"""Tests for multi-model LLM provider support."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from janus.config import GuardianModelConfig
from janus.llm.classifier import SecurityClassifier
from janus.llm.provider import LLMProvider
from janus.llm.providers import create_provider

# ── Mock provider for protocol testing ──────────────────────────────

class MockProvider:
    """A mock LLMProvider for testing."""

    def __init__(self, classify_result: dict[str, Any] | None = None, generate_result: str = ""):
        self._classify_result = classify_result or {"risk": 25, "reasoning": "test"}
        self._generate_result = generate_result

    async def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        return self._classify_result

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        return self._generate_result


# ── Protocol conformance ────────────────────────────────────────────

def test_mock_provider_satisfies_protocol():
    assert isinstance(MockProvider(), LLMProvider)


def test_anthropic_provider_satisfies_protocol():
    from janus.llm.providers.anthropic import AnthropicProvider
    # Can't instantiate without key, but can check class has required methods
    assert hasattr(AnthropicProvider, "classify")
    assert hasattr(AnthropicProvider, "generate")


# ── Factory ─────────────────────────────────────────────────────────

def test_factory_creates_anthropic():
    with patch("janus.llm.providers.anthropic.AnthropicProvider.__init__", return_value=None):
        provider = create_provider("anthropic", api_key="test-key")
        from janus.llm.providers.anthropic import AnthropicProvider
        assert isinstance(provider, AnthropicProvider)


def test_factory_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider("invalid_provider")


# ── SecurityClassifier with mock provider ───────────────────────────

async def test_classifier_with_mock_provider():
    provider = MockProvider(classify_result={"risk": 30, "reasoning": "moderate concern"})
    classifier = SecurityClassifier(client=provider)

    result = await classifier.classify_risk(
        agent_role="research",
        agent_name="test-bot",
        original_goal="Search for data",
        tool_name="execute_code",
        tool_input={"code": "print(1)"},
        session_history=[],
        current_risk_score=10.0,
    )
    assert result.risk == 30
    assert result.reasoning == "moderate concern"


async def test_classifier_drift_with_mock_provider():
    provider = MockProvider(
        classify_result={
            "drift_score": 0.7,
            "explanation": "high drift",
            "original_goal_summary": "research",
            "current_action_summary": "executing code",
        }
    )
    classifier = SecurityClassifier(client=provider)

    result = await classifier.classify_drift(
        original_goal="Research competitors",
        tool_name="execute_code",
        tool_input={"code": "import os"},
        conversation_history=[],
    )
    assert result.drift_score == 0.7


async def test_classifier_explain_with_mock_provider():
    provider = MockProvider(generate_result="The agent attempted to execute dangerous code.")
    classifier = SecurityClassifier(client=provider)

    result = await classifier.explain_trace(
        agent_name="test-bot",
        agent_role="research",
        tool_name="execute_code",
        tool_input={"code": "rm -rf /"},
        original_goal="Research",
        verdict="block",
        risk_score=85.0,
        drift_score=0.0,
        reasons=["dangerous payload"],
        itdr_signals=[],
    )
    assert "dangerous code" in result


# ── OpenAI provider with mocked SDK ────────────────────────────────

async def test_openai_provider_classify():
    from janus.llm.providers.openai_provider import OpenAIProvider

    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"risk": 45, "reasoning": "suspicious"}'
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = mock_client

    result = await provider.classify(
        system_prompt="test",
        user_prompt="test",
        model="gpt-4o-mini",
    )
    assert result["risk"] == 45


async def test_openai_provider_generate():
    from janus.llm.providers.openai_provider import OpenAIProvider

    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Generated text"
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = mock_client

    result = await provider.generate(
        system_prompt="test",
        user_prompt="test",
    )
    assert result == "Generated text"


# ── Ollama provider with mocked httpx ───────────────────────────────

async def test_ollama_provider_classify():
    from janus.llm.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider(base_url="http://localhost:11434")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "message": {"content": '{"risk": 20, "reasoning": "low risk"}'}
    }

    with patch("janus.llm.providers.ollama_provider.httpx.AsyncClient") as MockClient:
        client_instance = AsyncMock()
        client_instance.post = AsyncMock(return_value=mock_resp)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await provider.classify(
            system_prompt="test",
            user_prompt="test",
            model="llama3.2",
        )
        assert result["risk"] == 20
        # Verify format=json was requested
        call_kwargs = client_instance.post.call_args
        assert call_kwargs.kwargs.get("json", call_kwargs[1].get("json", {})).get("format") == "json"


async def test_ollama_provider_generate():
    from janus.llm.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider(base_url="http://localhost:11434")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "message": {"content": "Generated explanation text"}
    }

    with patch("janus.llm.providers.ollama_provider.httpx.AsyncClient") as MockClient:
        client_instance = AsyncMock()
        client_instance.post = AsyncMock(return_value=mock_resp)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await provider.generate(
            system_prompt="test",
            user_prompt="test",
        )
        assert result == "Generated explanation text"


# ── JSON parsing fallback ───────────────────────────────────────────

async def test_ollama_json_fallback():
    """Provider should extract JSON from text that contains extra content."""
    from janus.llm.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "message": {"content": 'Here is the analysis:\n{"risk": 50, "reasoning": "moderate"}\nDone.'}
    }

    with patch("janus.llm.providers.ollama_provider.httpx.AsyncClient") as MockClient:
        client_instance = AsyncMock()
        client_instance.post = AsyncMock(return_value=mock_resp)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await provider.classify(system_prompt="test", user_prompt="test")
        assert result["risk"] == 50


# ── Config ──────────────────────────────────────────────────────────

def test_guardian_model_config_defaults():
    config = GuardianModelConfig()
    assert config.provider == "anthropic"
    assert config.api_key == ""
    assert config.base_url == ""


def test_guardian_model_config_openai():
    config = GuardianModelConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
    )
    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"
