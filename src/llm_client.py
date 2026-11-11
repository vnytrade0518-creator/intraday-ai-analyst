import json
import streamlit as st
from src.config import ACTIVE_LLM_PROVIDER


def generate_analysis(system_prompt: str, user_prompt: str, json_mode: bool = True) -> dict | str:
    """
    Unified LLM router supporting Google Gemini and OpenAI.
    Controlled by ACTIVE_LLM_PROVIDER in src/config.py.
    """
    provider = ACTIVE_LLM_PROVIDER.lower().strip()

    if provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=st.secrets["llm"]["GEMINI_API_KEY"])
        model_name = st.secrets["llm"].get("GEMINI_MODEL", "gemini-2.5-flash")

        config_args = {
            "system_instruction": system_prompt,
            "temperature": 0.2,
        }
        if json_mode:
            config_args["response_mime_type"] = "application/json"

        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_args),
        )
        text = response.text
        return json.loads(text) if json_mode else text

    elif provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=st.secrets["llm"]["OPENAI_API_KEY"])
        model_name = st.secrets["llm"].get("OPENAI_MODEL", "gpt-4o-mini")

        kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content
        return json.loads(text) if json_mode else text

    else:
        raise ValueError(f"Unsupported ACTIVE_LLM_PROVIDER: {ACTIVE_LLM_PROVIDER}. Use 'gemini' or 'openai'.")
