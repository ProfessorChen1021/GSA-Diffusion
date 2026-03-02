# run_inference.py
import json
from pathlib import Path

from utils.common import (
    load_config,
    prepare_run_directory,
    dump_run_config,
)
from src.pipeline import GSADiffusionPipeline


def dummy_llm_fn(full_prompt: str) -> str:
    """
    full_prompt 是 planner 拼好的 system + user prompt，可以直接忽略。
    """
    data = {
        "entities": [
            {
                "phrase_entity": "cat",
                "phrase_ent_attr": "a fluffy red cat",
            },
            {
                "phrase_entity": "rabbit",
                "phrase_ent_attr": "a blue rabbit",
            },
        ]
    }
    return json.dumps(data, ensure_ascii=False)

def main() -> None:
    # 1) 载入配置
    cfg, config_dir = load_config()

    # 2) 准备本次运行的输出目录
    run_dir = prepare_run_directory(cfg["paths"]["output_root"], config_dir)
    dump_run_config(cfg, run_dir)

    # 3) 构建 GSADiffusionPipeline，llm_fn
    pipeline = GSADiffusionPipeline(
        cfg=cfg,
        config_dir=config_dir,
        llm_fn=dummy_llm_fn,
    )

    # 4) 跑数据集（目前 datasets 里就一个数字子目录）
    pipeline.run_dataset(outputs_root=Path(run_dir))


if __name__ == "__main__":
    main()

