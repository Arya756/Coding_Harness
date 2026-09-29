"""Native Google Gemini v1beta provider."""

from __future__ import annotations

import json
from typing import Any, Optional

import requests

from .llm import LLMProvider, LlmResponse, estimate_cost
from .models import ModelHTTPError, ServiceUnavailableError, _action


class GeminiProvider(LLMProvider):
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @staticmethod
    def _to_gemini_messages(messages: list) -> tuple[Optional[str], list]:
        system_instruction = None
        contents = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                system_instruction = (system_instruction + "\n" + content) if system_instruction else content
                continue
            gem_role = "model" if role == "assistant" else "user"
            contents.append({"role": gem_role, "parts": [{"text": content}]})
        return system_instruction, contents

    @staticmethod
    def _build_json_schema_variants(tools: list) -> dict:
        """Build a responseJsonSchema forcing {name, arguments, goal}."""
        variants = []
        for t in tools:
            variants.append({
                "type": "object",
                "properties": {
                    "name": {"const": t.get("name")},
                    "arguments": {"type": "object"},
                    "goal": {"type": "string"},
                },
                "required": ["name"],
            })
        return {
            "anyOf": variants
            if variants
            else [{
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                    "goal": {"type": "string"},
                },
                "required": ["name"],
            }]
        }

    def chat(self, messages: list, max_tokens: int = 4096, temperature: float = 0.7,
              tools: Optional[list] = None) -> LlmResponse:

        def _do_call():
            system_instruction, contents = self._to_gemini_messages(messages)
            payload: dict[str, Any] = {
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                },
            }
            if system_instruction:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
            if tools:
                payload["generationConfig"]["responseMimeType"] = "application/json"
                payload["generationConfig"]["responseJsonSchema"] = self._build_json_schema_variants(tools)

            url = f"{self.BASE_URL}/models/{self._model}:generateContent?key={self.api_key}"
            try:
                resp = requests.post(url, data=json.dumps(payload),
                                      headers={"Content-Type": "application/json"}, timeout=120)
            except requests.exceptions.Timeout as exc:
                raise ServiceUnavailableError(str(exc)) from exc
            except requests.exceptions.ConnectionError as exc:
                raise ServiceUnavailableError(str(exc)) from exc

            if resp.status_code == 429:
                raise ModelHTTPError(429, resp.text)
            if resp.status_code >= 500:
                raise ModelHTTPError(resp.status_code, resp.text)
            if resp.status_code >= 400:
                raise ModelHTTPError(resp.status_code, resp.text)

            data = resp.json()
            candidates = data.get("candidates", [])
            text = ""
            tool_calls = []
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        tool_calls.append({"name": fc.get("name", ""), "args": fc.get("args", {})})
                    elif "text" in part:
                        text += part["text"]

            if not tool_calls and text:
                act = _action(text)
                if act:
                    tool_calls.append({"name": act.name, "args": act.arguments})

            usage = data.get("usageMetadata", {})
            in_tok = usage.get("promptTokenCount", 0)
            out_tok = usage.get("candidatesTokenCount", 0)
            cost = estimate_cost(self._model, in_tok, out_tok)
            return LlmResponse(
                text=text, input_tokens=in_tok, output_tokens=out_tok,
                cost_usd=cost, model=self._model, tool_calls=tool_calls,
            )

        return self._with_retry(_do_call)
