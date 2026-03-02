# src/llm_client.py
# -*- coding: utf-8 -*-
"""
LLM 客户端封装。
可以在这里调用外部 API（如 OpenAI / DeepSeek / 本地 LLM 服务等），
只需保证 call_llm_api(prompt: str) -> str 返回**纯文本字符串**即可。
"""

from __future__ import annotations
import os
from typing import Optional
from openai import OpenAI

client = OpenAI(
    api_key="00000",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def call_llm_api(prompt: str, model: Optional[str] = None) -> str:
    """
    包一层，给 planner 调用。
    prompt：已经是构造好的“大提示词”（里面包含原始用户 prompt）。
    """
    resp = client.chat.completions.create(
        model=model or "qwen-plus",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0, 
    )
    print(resp.choices[0].message.content)

    return resp.choices[0].message.content
