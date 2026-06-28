"""
LLM provider abstraction (BYOK — bring your own key).

OpenAI, Groq, and Google Gemini all speak the OpenAI chat-completions API, so
one async path works for them with a per-provider base URL. Anthropic uses its
own Messages API, handled by a dedicated path. API keys are supplied per request
by the user (browser localStorage -> X-LLM-* headers) and are never stored,
cached, or logged server-side — the server only relays the call.
"""

from __future__ import annotations

import httpx

# Provider catalog. `api` selects the request shape ("openai" | "anthropic");
# `pricing` is "free" (free tier available) or "paid"; `json_mode` marks
# providers that honor an OpenAI-style JSON response_format.
PROVIDERS: dict[str, dict] = {
    "openai": {
        "label": "OpenAI",
        "api": "openai",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"],
        "key_hint": "sk-...",
        "key_url": "https://platform.openai.com/api-keys",
        "pricing": "paid",
        "json_mode": True,
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "api": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-opus-4-8"],
        "key_hint": "sk-ant-...",
        "key_url": "https://console.anthropic.com/settings/keys",
        "pricing": "paid",
        "json_mode": False,
    },
    "gemini": {
        "label": "Google Gemini",
        "api": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        "key_hint": "AIza...",
        "key_url": "https://aistudio.google.com/app/apikey",
        "pricing": "free",
        "json_mode": False,
    },
    "groq": {
        "label": "Groq",
        "api": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b"],
        "key_hint": "gsk_...",
        "key_url": "https://console.groq.com/keys",
        "pricing": "free",
        "json_mode": True,
    },
}


class LLMError(Exception):
    """Carries an HTTP status so the API can surface a clean error to the UI."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(message)


def public_registry() -> dict:
    """Provider/model catalog for the frontend Settings dropdowns. No secrets."""
    return {
        "providers": [
            {
                "id": pid,
                "label": p["label"],
                "models": p["models"],
                "key_hint": p["key_hint"],
                "key_url": p["key_url"],
                "pricing": p["pricing"],
            }
            for pid, p in PROVIDERS.items()
        ]
    }


def validate(provider: str, model: str, api_key: str) -> None:
    if provider not in PROVIDERS:
        raise LLMError(400, f"Unknown provider '{provider}'. Choose: {', '.join(PROVIDERS)}.")
    if not api_key:
        raise LLMError(
            401, "Missing API key. Add your key in the Settings tab (stored only in your browser)."
        )
    if model and model not in PROVIDERS[provider]["models"]:
        raise LLMError(400, f"Model '{model}' is not available for {PROVIDERS[provider]['label']}.")


def _provider_error(label: str, r: httpx.Response) -> LLMError:
    if r.status_code == 401:
        return LLMError(401, f"{label} rejected the API key (401). Check it in Settings.")
    detail = ""
    try:
        body = r.json()
        detail = (body.get("error") or {}).get("message", "") if isinstance(body, dict) else ""
    except Exception:
        detail = r.text[:200]
    return LLMError(r.status_code, f"{label} error: {detail or r.status_code}")


async def _openai_chat(cfg, model, api_key, messages, temperature, max_tokens, json_mode) -> str:
    payload: dict = {
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }
    if json_mode and cfg.get("json_mode"):
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{cfg['base_url']}/chat/completions", json=payload, headers=headers
            )
    except httpx.RequestError as exc:
        raise LLMError(502, f"Could not reach {cfg['label']}: {exc}") from exc
    if r.status_code >= 400:
        raise _provider_error(cfg["label"], r)
    try:
        return r.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(502, f"Malformed response from {cfg['label']}.") from exc


async def _anthropic_chat(cfg, model, api_key, messages, temperature, max_tokens) -> str:
    # Anthropic takes the system prompt as a top-level field and only
    # user/assistant turns in messages.
    system = " ".join(m["content"] for m in messages if m.get("role") == "system")
    convo = [{"role": m["role"], "content": m["content"]}
             for m in messages if m.get("role") in ("user", "assistant")]
    payload: dict = {
        "model": model, "max_tokens": max_tokens,
        "temperature": temperature, "messages": convo,
    }
    if system:
        payload["system"] = system
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{cfg['base_url']}/messages", json=payload, headers=headers)
    except httpx.RequestError as exc:
        raise LLMError(502, f"Could not reach {cfg['label']}: {exc}") from exc
    if r.status_code >= 400:
        raise _provider_error(cfg["label"], r)
    try:
        return r.json()["content"][0]["text"] or ""
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(502, f"Malformed response from {cfg['label']}.") from exc


async def chat(
    provider: str,
    model: str,
    api_key: str,
    messages: list[dict],
    *,
    temperature: float = 0.2,
    max_tokens: int = 900,
    json_mode: bool = False,
) -> str:
    """Relay one chat completion to the chosen provider and return the text.

    Raises LLMError (with an HTTP status) on bad input or provider failure. The
    API key never appears in any raised message or log line.
    """
    validate(provider, model, api_key)
    cfg = PROVIDERS[provider]
    model = model or cfg["models"][0]
    if cfg["api"] == "anthropic":
        return await _anthropic_chat(cfg, model, api_key, messages, temperature, max_tokens)
    return await _openai_chat(cfg, model, api_key, messages, temperature, max_tokens, json_mode)
