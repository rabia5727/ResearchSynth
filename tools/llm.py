"""LLM provider abstraction.

Every agent calls generate_json() instead of talking to an SDK directly.
That's what lets the team switch Gemini <-> Groq with one env var (see the
plan, section 2's Day-1 smoke test), and lets anyone build and test their
agent with LLM_PROVIDER=mock before a single real API key exists.
"""
from __future__ import annotations

import json
import os
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when a provider call fails or returns unparseable JSON."""


def generate_json(prompt: str, schema: Type[T], *, system: str | None = None) -> T:
    """Call the configured LLM provider and parse its response into `schema`.

    Provider is chosen by the LLM_PROVIDER env var: "gemini" | "groq" | "mock".
    Raises LLMError on any failure - callers decide whether to retry, skip,
    or fail loudly.
    """
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    if provider == "gemini":
        raw = _call_gemini(prompt, schema, system)
    elif provider == "groq":
        raw = _call_groq(prompt, schema, system)
    elif provider == "mock":
        raw = _call_mock(schema)
    else:
        raise LLMError(f"Unknown LLM_PROVIDER: {provider!r} (use gemini | groq | mock)")

    try:
        return schema.model_validate_json(raw)
    except Exception as exc:  # noqa: BLE001 - normalize every failure to LLMError
        raise LLMError(
            f"{provider} returned invalid JSON for {schema.__name__}: {exc}\nraw={raw[:500]}"
        ) from exc


def _call_gemini(prompt: str, schema: Type[BaseModel], system: str | None) -> str:
    try:
        from google import genai
    except ImportError as exc:
        raise LLMError("google-genai is not installed - `pip install google-genai`") from exc

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    response = client.models.generate_content(
        model=model,
        contents=f"{system}\n\n{prompt}" if system else prompt,
        config={"response_mime_type": "application/json", "response_schema": schema},
    )
    return response.text


def _call_groq(prompt: str, schema: Type[BaseModel], system: str | None) -> str:
    try:
        from groq import Groq
    except ImportError as exc:
        raise LLMError("groq is not installed - `pip install groq`") from exc

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not set")

    client = Groq(api_key=api_key)
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    schema_hint = json.dumps(schema.model_json_schema())

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": f"{prompt}\n\nRespond with ONLY valid JSON matching this schema:\n{schema_hint}",
        }
    )
    completion = client.chat.completions.create(
        model=model, messages=messages, response_format={"type": "json_object"}
    )
    return completion.choices[0].message.content


def _call_mock(schema: Type[BaseModel]) -> str:
    """Deterministic stand-in for building/testing an agent with no API key.

    Walks the schema's JSON Schema and fills every field with a placeholder
    value of the right type/shape, so any agent's schema works here without
    per-agent mock code.
    """
    json_schema = schema.model_json_schema()
    defs = json_schema.get("$defs", {})
    return json.dumps(_dummy_object(json_schema, defs))


def _resolve(node: dict, defs: dict) -> dict:
    if "$ref" in node:
        return defs.get(node["$ref"].split("/")[-1], {})
    return node


def _dummy_object(node: dict, defs: dict) -> dict:
    props = node.get("properties", {})
    return {name: _dummy_value(sub, defs, name) for name, sub in props.items()}


def _dummy_value(node: dict, defs: dict, name: str = "") -> object:
    node = _resolve(node, defs)
    if "enum" in node:
        return node["enum"][0]
    if "anyOf" in node:
        for option in node["anyOf"]:
            option = _resolve(option, defs)
            if option.get("type") != "null":
                return _dummy_value(option, defs, name)
        return None

    kind = node.get("type")
    if kind == "string":
        return f"[mock:{name}]"
    if kind == "integer":
        return 0
    if kind == "number":
        return 0.0
    if kind == "boolean":
        return False
    if kind == "array":
        return [_dummy_value(node.get("items", {}), defs, name)]
    if kind == "object" or "properties" in node:
        return _dummy_object(node, defs)
    return None
