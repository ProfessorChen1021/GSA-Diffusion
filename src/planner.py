"""Break a prompt into entity-level descriptions for grounding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


LLMFn = Callable[[str], str]


@dataclass
class PlannerConfig:
    mode: str = "rule"  # "rule" or "llm"


class PromptEntityPlanner:
    """Turn a full prompt into the smaller pieces used downstream."""

    def __init__(self, cfg: PlannerConfig, llm_fn: Optional[LLMFn] = None) -> None:
        self.mode = cfg.mode
        self.llm_fn = llm_fn
        if self.mode == "llm" and self.llm_fn is None:
            raise RuntimeError("Planner mode is 'llm' but llm_fn is missing.")

    def plan(self, prompt: str, sample_root: Path) -> List[Dict[str, Any]]:
        sample_root = Path(sample_root)
        cache_path = sample_root / "prompt_decoupled.json"

        if cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                cached_entities = self._from_cached_json(data)
                if cached_entities:
                    return cached_entities
            except Exception:
                # Ignore a broken cache and rebuild it from the prompt.
                pass

        if self.mode == "rule":
            entities = self._plan_by_rule(prompt)
        elif self.mode == "llm":
            entities = self._plan_by_llm(prompt)
        else:
            raise ValueError(f"Unknown planner mode: {self.mode}")

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
        except Exception:
            pass

        return entities

    def _from_cached_json(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        entities_raw = data.get("entities", []) if isinstance(data, dict) else []
        entities: List[Dict[str, Any]] = []
        for item in entities_raw:
            if not isinstance(item, dict):
                continue
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

        return entities

    def _build_llm_prompt(self, prompt: str) -> str:
        template = f"""
You are helping a text-to-image system prepare grounding targets.

Read the prompt and list the visual entities that should be grounded.
For each entity, return:
- name: a short English noun phrase (for example: "dog", "red car")
- description: only the entity's own visible attributes, such as color, material, texture, or appearance

Please keep a few rules in mind:
- Do not describe the background or scene.
- Do not mention position or relationships to other objects.
- Do not add explanations outside the JSON.
- Return valid JSON only.

Use this format:
{{
  "prompt": "<original prompt>",
  "entities": [
    {{"name": "<entity_1>", "description": "<description_1>"}},
    {{"name": "<entity_2>", "description": "<description_2>"}}
  ]
}}

User prompt:
"{prompt}"
"""
        return template.strip()

    @staticmethod
    def _extract_json_text(text: str) -> str:
        cleaned = text.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1]
        return cleaned

    def _plan_by_llm(self, prompt: str) -> List[Dict[str, Any]]:
        if self.llm_fn is None:
            raise RuntimeError("LLM planner requires llm_fn.")

        raw = self.llm_fn(self._build_llm_prompt(prompt))
        json_text = self._extract_json_text(raw)

        try:
            data = json.loads(json_text)
        except Exception as exc:
            preview = str(raw).strip().replace("\n", " ")[:300]
            raise RuntimeError(f"LLM JSON parse error: {exc}; response preview: {preview}") from exc

        entities = self._from_cached_json(data)
        if entities:
            return entities

        return [
            {
                "phrase_entity": "scene",
                "phrase_ent_attr": prompt,
                "mask_tensor": None,
            }
        ]
