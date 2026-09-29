"""LLM provider abstraction: providers, factory, pricing, shared retry logic."""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import requests

from .models import ModelHTTPError, ProtocolError, ServiceUnavailableError, _action


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class LlmResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    tool_calls: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pricing (per 1M tokens, USD)
# ---------------------------------------------------------------------------

_PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    "deepseek-chat": (0.14, 0.28),
    "qwen-turbo": (0.06, 0.20),
    "qwen-plus": (0.10, 0.30),
    "qwen-max": (0.50, 1.00),
    "llama-3": (0.65, 2.75),
    "mixtral": (0.40, 1.50),
    "groq-llama": (0.05, 0.19),
    "gemini-2.5-flash": (0.075, 0.30),
    "gemini-2.5-pro": (1.25, 5.00),
    "default": (0.50, 1.50),
}


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    key = model
    if key not in _PRICING:
        # try prefix match
        key = next((k for k in _PRICING if model.startswith(k)), "default")
    in_price, out_price = _PRICING.get(key, _PRICING["default"])
    return (in_tok / 1_000_000) * in_price + (out_tok / 1_000_000) * out_price


# ---------------------------------------------------------------------------
# Base provider
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def chat(self, messages: list, max_tokens: int = 4096, temperature: float = 0.7,
              tools: Optional[list] = None) -> LlmResponse:
        ...

    @staticmethod
    def extract_tool_call(response_text: str) -> Optional[ToolCall]:
        act = _action(response_text)
        if act is None:
            return None
        return ToolCall(name=act.name, args=act.arguments)

    def _with_retry(self, fn: Callable[[], Any], max_attempts: int = 3, base_delay: float = 1.0) -> Any:
        last_exc = None
        for attempt in range(max_attempts):
            try:
                return fn()
            except ServiceUnavailableError as exc:
                last_exc = exc
                if attempt == max_attempts - 1:
                    raise
                time.sleep(base_delay * (2 ** attempt))
            except ModelHTTPError as exc:
                last_exc = exc
                if exc.status in (429, 500, 502, 503, 504) and attempt < max_attempts - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError("retry loop exited unexpectedly")


# ---------------------------------------------------------------------------
# OpenAI-compatible base
# ---------------------------------------------------------------------------

class _OpenAICompatibleProvider(LLMProvider):
    """Shared implementation for any OpenAI chat-completions compatible API."""

    def __init__(self, api_key: str, model: str, base_url: str, extra_headers: Optional[dict] = None):
        self.api_key = api_key
        self._model = model
        self.base_url = base_url.rstrip("/")
        self.extra_headers = extra_headers or {}

    @property
    def model_name(self) -> str:
        return self._model

    def _max_tokens_cap(self, requested: int) -> int:
        return requested

    def chat(self, messages: list, max_tokens: int = 4096, temperature: float = 0.7,
              tools: Optional[list] = None) -> LlmResponse:

        def _do_call():
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            headers.update(self.extra_headers)
            payload: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": self._max_tokens_cap(max_tokens),
                "temperature": temperature,
            }
            if tools:
                payload["tools"] = [
                    {"type": "function", "function": t} for t in tools
                ]
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=120,
                )
            except requests.exceptions.Timeout as exc:
                raise ServiceUnavailableError(str(exc)) from exc
            except requests.exceptions.ConnectionError as exc:
                raise ServiceUnavailableError(str(exc)) from exc

            if resp.status_code in (429, 502, 503, 504):
                raise ModelHTTPError(resp.status_code, resp.text)
            if resp.status_code >= 400:
                raise ModelHTTPError(resp.status_code, resp.text)

            data = resp.json()
            choice = data["choices"][0]
            message = choice.get("message", {})
            text = message.get("content") or ""
            tool_calls = []
            for tc in message.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({"name": fn.get("name", ""), "args": args})

            usage = data.get("usage", {}) or {}
            in_tok = usage.get("prompt_tokens", 0)
            out_tok = usage.get("completion_tokens", 0)
            cost = estimate_cost(self._model, in_tok, out_tok)
            return LlmResponse(
                text=text, input_tokens=in_tok, output_tokens=out_tok,
                cost_usd=cost, model=self._model, tool_calls=tool_calls,
            )

        return self._with_retry(_do_call)


class OpenaiProvider(_OpenAICompatibleProvider):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model, "https://api.openai.com/v1")


class OpenRouterProvider(_OpenAICompatibleProvider):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model, "https://openrouter.ai/api/v1")


class OllamaProvider(_OpenAICompatibleProvider):
    def __init__(self, api_key: str, model: str, host: Optional[str] = None):
        host = host or os.environ.get("HARNESS_LLM_HOST", "http://localhost:11434")
        super().__init__(api_key or "ollama", model, f"{host.rstrip('/')}/v1")


class GroqProvider(_OpenAICompatibleProvider):
    def __init__(self, api_key: str, model: str):
        base = os.environ.get("HARNESS_LLM_GROQ_BASE", "https://api.groq.com/openai/v1")
        super().__init__(api_key, model, base)

    def _max_tokens_cap(self, requested: int) -> int:
        # Groq free tier cap. This restriction lives only here.
        return min(requested, 900)


class DeepSeekProvider(_OpenAICompatibleProvider):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model, "https://api.deepseek.com/v1")


class QwenProvider(_OpenAICompatibleProvider):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model, "https://dashscope.aliyuncs.com/compatible-mode/v1")


class GenericProvider(_OpenAICompatibleProvider):
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        base_url = base_url or os.environ.get("HARNESS_LLM_BASE_URL") or os.environ.get(
            "HARNESS_LLM_CURSOR_BASE", "https://api.cursor.com/v1"
        )
        super().__init__(api_key, model, base_url)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _provider_map():
    from .llm_gemini import GeminiProvider  # lazy import to avoid circular deps
    return {
        "openai": OpenaiProvider,
        "openrouter": OpenRouterProvider,
        "ollama": OllamaProvider,
        "groq": GroqProvider,
        "deepseek": DeepSeekProvider,
        "qwen": QwenProvider,
        "gemini": GeminiProvider,
        "cursor": GenericProvider,
        "generic": GenericProvider,
    }


def create_provider(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    ollama_host: Optional[str] = None,
) -> LLMProvider:
    provider = provider or os.environ.get("HARNESS_LLM_PROVIDER", "openai")
    api_key = (
        api_key
        or os.environ.get("AI_API_KEY")
        or os.environ.get("HARNESS_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    model = model or os.environ.get("HARNESS_LLM_MODEL") or os.environ.get("HARNESS_MODEL", "gpt-4o-mini")
    base_url = base_url or os.environ.get("HARNESS_LLM_BASE_URL")

    mapping = _provider_map()
    provider = (provider or "openai").lower()
    cls = mapping.get(provider)
    if cls is None:
        raise ValueError(f"Unknown LLM provider: {provider}")

    if provider == "ollama":
        return cls(api_key, model, host=ollama_host)
    if provider in ("cursor", "generic"):
        return cls(api_key, model, base_url=base_url)
    return cls(api_key, model)
