import json
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


def _call_gemini(system_prompt: str, user_prompt: str, model: str, json_mode: bool) -> str:
    api_key = st.secrets["llm"]["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)

    config_kwargs: Dict[str, Any] = {
        "system_instruction": system_prompt,
        "temperature": 0.2,
    }
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return response.text or "{}"


def _call_openai(system_prompt: str, user_prompt: str, model: str, json_mode: bool) -> str:
    api_key = st.secrets["llm"]["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key, max_retries=0)

    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content or "{}"


def generate_analysis(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = True,
    provider: str | None = None,
) -> Dict[str, Any]:
    """Unified, rate-limited LLM router supporting Gemini and OpenAI."""
    active = (provider or ACTIVE_LLM_PROVIDER).lower()
    if active not in LLM_LIMITS:
        raise ValueError(f"Unsupported LLM provider: {active}")

    model = LLM_LIMITS[active]["model"]
    limiter = _get_limiters()[active]
    est_tokens = _estimate_tokens(system_prompt, user_prompt)
    limiter.acquire(estimated_tokens=est_tokens)

    if active == "gemini":
        raw_text = execute_with_backoff(_call_gemini, system_prompt, user_prompt, model, json_mode)
    else:
        raw_text = execute_with_backoff(_call_openai, system_prompt, user_prompt, model, json_mode)

    if not json_mode:
        return {"raw_text": raw_text}

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    return json.loads(cleaned)
