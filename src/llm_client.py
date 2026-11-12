from typing import Any, Dict

import streamlit as st
from google import genai
from google.genai import types
from openai import OpenAI

from src.config import ACTIVE_LLM_PROVIDER, LLM_LIMITS
from src.rate_limiter import TokenBucketRateLimiter, execute_with_backoff


@st.cache_resource
def _get_limiters() -> Dict[str, TokenBucketRateLimiter]:
    return {
        provider: TokenBucketRateLimiter(
            max_rpm=cfg["max_rpm"],
            max_tpm=cfg["max_tpm"],
            max_rpd=cfg["max_rpd"],
        )
        for provider, cfg in LLM_LIMITS.items()
    }


def get_quota_status(provider: str | None = None) -> Dict[str, int]:
    active = provider or ACTIVE_LLM_PROVIDER
    return _get_limiters()[active].status()


def _estimate_tokens(system_prompt: str, user_prompt: str) -> int:
    return max(500, (len(system_prompt) + len(user_prompt)) // 4 + 800)


def generate_analysis(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = True,
    provider_override: str | None = None,
) -> str:
    """Unified, rate-limited LLM router supporting Gemini 3.8 Flash and GPT-5.4-mini."""
    active = (provider_override or ACTIVE_LLM_PROVIDER).lower()
    if active not in LLM_LIMITS:
        raise ValueError(f"Unsupported LLM provider: {active}")

    cfg = LLM_LIMITS[active]
    model_name = st.secrets.get("llm", {}).get("MODEL") or cfg["model"]
    limiter = _get_limiters()[active]
    est_tokens = _estimate_tokens(system_prompt, user_prompt)

    limiter.acquire(estimated_tokens=est_tokens)

    if active == "gemini":
        api_key = st.secrets["llm"]["GEMINI_API_KEY"]
        client = genai.Client(
            api_key=api_key,
            http_options={"timeout": 30_000},
        )
        config_kwargs: Dict[str, Any] = {
            "system_instruction": system_prompt,
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        def _call_gemini() -> str:
            resp = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return resp.text or "{}"

        return execute_with_backoff(_call_gemini)

    if active == "openai":
        api_key = st.secrets["llm"]["OPENAI_API_KEY"]
        client = OpenAI(api_key=api_key, max_retries=0, timeout=30.0)
        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        def _call_openai() -> str:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or "{}"

        return execute_with_backoff(_call_openai)

    raise RuntimeError(f"Unhandled provider: {active}")
