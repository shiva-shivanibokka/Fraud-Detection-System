"""
LLM provider abstraction (BYOK — bring your own key).

OpenAI and Groq both speak the OpenAI chat-completions API, so one async client
works for both with a per-provider base URL. API keys are supplied *per request*
by the user (browser localStorage -> X-LLM-* headers) and are never stored,
cached, or logged server-side. The server only relays the call.
"""

from __future__ import annotations

import httpx

# Provider -> available models. Models are an explicit allow-list so a stray /
# malicious model string can't be relayed to the provider on the user's key.
PROVIDERS: dict[str, dict] = {
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"],
        "key_hint": "sk-...",
        "key_url": "https://platform.openai.com/api-keys",
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b"],
        "key_hint": "gsk_...",
        "key_url": "https://console.groq.com/keys",
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
    """Relay one chat-completion to the chosen provider and return the text.

    Raises LLMError (with an HTTP status) on bad input or provider failure. The
    API key never appears in any raised message or log line.
    """
    validate(provider, model, api_key)
    cfg = PROVIDERS[provider]
    model = model or cfg["models"][0]

    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{cfg['base_url']}/chat/completions", json=payload, headers=headers
            )
    except httpx.RequestError as exc:
        raise LLMError(502, f"Could not reach {cfg['label']}: {exc}") from exc

    if r.status_code == 401:
        raise LLMError(401, f"{cfg['label']} rejected the API key (401). Check it in Settings.")
    if r.status_code >= 400:
        detail = ""
        try:
            detail = (r.json().get("error") or {}).get("message", "")
        except Exception:
            detail = r.text[:200]
        raise LLMError(r.status_code, f"{cfg['label']} error: {detail or r.status_code}")

    try:
        return r.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(502, f"Malformed response from {cfg['label']}.") from exc
