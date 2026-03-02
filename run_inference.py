# run_inference.py
# -*- coding: utf-8 -*-

from utils.common import load_config
from src.pipeline import GSADiffusionPipeline
from src.llm_client import call_llm_api


def llm_fn(prompt: str) -> str:
    """
    LLM wrapper used by the planner; receives the full planner prompt (includes user prompt).
    """
    print("[LLM_FN] called with prompt length:", len(prompt))
    resp = call_llm_api(prompt)
    print("[LLM_FN] got response length:", len(resp))
    return resp


def main():
    cfg, config_dir = load_config()
    pipeline = GSADiffusionPipeline(cfg=cfg, config_dir=config_dir, llm_fn=llm_fn)
    pipeline.run_dataset()


if __name__ == "__main__":
    main()
