"""Chat client used by the planner."""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


def _build_client() -> OpenAI:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set LLM_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY.")

    base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm_api(prompt: str, model: Optional[str] = None) -> str:
    """Send one prompt and return the model's text reply."""
    client = _build_client()
    response = client.chat.completions.create(
        model=model or os.getenv("LLM_MODEL", "qwen-plus"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()
