# src/planner.py
# -*- coding: utf-8 -*-
"""
Prompt 实体解耦模块（Planner）。

- 支持 mode="rule" 和 mode="llm"
- 优先读 datasets/<sample_id>/prompt_decoupled.json
- 没有缓存时，按 mode 调用规则或 LLM
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


LLMFn = Callable[[str], str]


@dataclass
class PlannerConfig:
    mode: str = "rule"  # "rule" 或 "llm"


class PromptEntityPlanner:
    """
    plan(prompt, sample_root) -> List[Dict]:
        [
            {
                "phrase_entity": "cat",
                "phrase_ent_attr": "a fluffy red cat",
                "mask_tensor": None,
            },
            ...
        ]
    """

    def __init__(self, cfg: PlannerConfig, llm_fn: Optional[LLMFn] = None) -> None:
        self.mode = cfg.mode
        self.llm_fn = llm_fn

        print(f"[Planner] init mode={self.mode}, llm_fn is None? {self.llm_fn is None}")

        if self.mode == "llm" and self.llm_fn is None:
            raise RuntimeError("Planner 配置为 mode='llm'，但未提供 llm_fn。")

    # ------------ 公共入口 ------------

    def plan(self, prompt: str, sample_root: Path) -> List[Dict[str, Any]]:
        sample_root = Path(sample_root)
        cache_path = sample_root / "prompt_decoupled.json"

        print(f"[Planner] plan() called for sample_root={sample_root}")
        print(f"[Planner] cache_path={cache_path}")

        # 1) 读缓存
        if cache_path.exists():
            print("[Planner] cache exists, trying to load...")
            try:
                with cache_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                entities = self._from_cached_json(data)
                if entities:
                    print("[Planner] Loaded entities from cache.")
                    return entities
            except Exception as e:
                print(f"[Planner] Failed to load cache: {e}")

        # 2) 没缓存 -> 按 mode 解耦
        print(f"[Planner] No valid cache, running mode={self.mode}")
        if self.mode == "rule":
            entities = self._plan_by_rule(prompt)
        elif self.mode == "llm":
            entities = self._plan_by_llm(prompt)
        else:
            raise ValueError(f"Unknown planner mode: {self.mode}")

        # 3) 写缓存
        cache_obj = {
            "prompt": prompt,
            "entities": [
                {
                    "name": ent["phrase_entity"],
                    "description": ent.get("phrase_ent_attr", ent["phrase_entity"]),
                }
                for ent in entities
            ],
        }
        try:
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(cache_obj, f, ensure_ascii=False, indent=2)
            print(f"[Planner] Saved decoupled prompt to: {cache_path}")
        except Exception as e:
            print(f"[Planner] Failed to save cache {cache_path}: {e}")

        return entities

    # ------------ 缓存 JSON -> entities ------------

    def _from_cached_json(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        ents_raw = data.get("entities", [])
        entities: List[Dict[str, Any]] = []
        for item in ents_raw:
            name = item.get("name")
            desc = item.get("description") or name
            if not name:
                continue
            entities.append(
                {
                    "phrase_entity": str(name),
                    "phrase_ent_attr": str(desc),
                    "mask_tensor": None,
                }
            )
        return entities

    # ------------ 规则模式（调试用） ------------

    def _plan_by_rule(self, prompt: str) -> List[Dict[str, Any]]:
        prompt_lower = prompt.lower()
        entities: List[Dict[str, Any]] = []

        if "cat" in prompt_lower:
            entities.append(
                {
                    "phrase_entity": "cat",
                    "phrase_ent_attr": "a red cat" if "red" in prompt_lower else "a cat",
                    "mask_tensor": None,
                }
            )

        if "rabbit" in prompt_lower or "bunny" in prompt_lower:
            entities.append(
                {
                    "phrase_entity": "rabbit",
                    "phrase_ent_attr": "a blue rabbit" if "blue" in prompt_lower else "a rabbit",
                    "mask_tensor": None,
                }
            )

        if not entities:
            entities.append(
                {
                    "phrase_entity": "scene",
                    "phrase_ent_attr": prompt,
                    "mask_tensor": None,
                }
            )

        print(f"[Planner] Rule-based entities: {entities}")
        return entities

    # ------------ LLM 模式 ------------

    def _build_llm_prompt(self, prompt: str) -> str:
        """
        只抽取实体自身属性（颜色 / 材质 / 外观 / 姿态），不包含位置 / 背景 / 其它物体。
        """
        template = f"""
   You are a vision-language planner for text-to-image generation.

    Task:
    Given a natural language description of an image, extract all the "physical entities" (objects, animals, people, etc.) mentioned in the prompt.

    For each entity you MUST output:
      - "name": a **shortest** English word (e.g., "dog", "sheep")
      - "description": a short phrase that only describes the "intrinsic attributes" of the entity
          - allowed: color, material, texture, shape, style, appearance
          - NOT allowed: descriptions of the background, scene, location, other objects, or interactions with other entities

    IMPORTANT constraints for every "description":
      - Describe with the **complete name** and in accordance with grammar!!!
      - Do not mention the location of the entity (excluding location or scene information).
        - Forbidden: Prohibit the use of expressions such as "in the room", "on the table", "in the sky", "next to a tree", "near the window", "on the road", etc.
      - Do not mention other entities or objects.
        - Forbidden: "next to another person", "beside a dog", etc.
      - Focus only on how this entity itself looks or behaves.
        - Allowed: color, material, texture, shape, style, appearance.

    Output format:
    Return a single JSON object in the following format:

    {{
      "prompt": "<copy of the original prompt>",
      "entities": [
        {{"name": "<entity_name_1>", "description": "<description_of_entity_1>"}},
        {{"name": "<entity_name_2>", "description": "<description_of_entity_2>"}}
      ]
    }}

    Requirements:
    - The "entities" list must not be empty.
    - Use strictly valid JSON.
    - Do NOT add any explanation or text outside the JSON.
    - Names only without descriptions are not included.
    - Describe with the **complete name** and in accordance with grammar!!!
    User prompt:
    \"\"\"{prompt}\"\"\"
"""
        return template.strip()

    def _plan_by_llm(self, prompt: str) -> List[Dict[str, Any]]:
        print("[Planner] _plan_by_llm called")
        if self.llm_fn is None:
            raise RuntimeError("LLM planner requires llm_fn, but got None.")

        llm_prompt = self._build_llm_prompt(prompt)
        raw = self.llm_fn(llm_prompt)
        text = raw.strip()

        # 【增强】：去除可能存在的 Markdown 代码块标记
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # 寻找第一个 { 和最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start: end + 1]

        try:
            data = json.loads(text)
        except Exception as e:
            print(f"[Planner] Failed to parse LLM JSON, raw output:\n{raw}")
            raise RuntimeError(f"LLM JSON parse error: {e}")

        entities = self._from_cached_json(data)
        if not entities:
            entities = [
                {
                    "phrase_entity": "scene",
                    "phrase_ent_attr": prompt,
                    "mask_tensor": None,
                }
            ]
        print(f"[Planner] LLM-based entities: {entities}")
        return entities
