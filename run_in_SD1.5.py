from src.pipeline import GSADiffusionPipeline
from utils.common import describe_public_release_dataset, load_config


def llm_fn(prompt: str) -> str:
    from src.llm_client import call_llm_api

    return call_llm_api(prompt)


def main() -> None:
    cfg, config_dir = load_config()
    public_release_note = describe_public_release_dataset(cfg, config_dir)
    if public_release_note:
        print(public_release_note)
        return

    pipeline = GSADiffusionPipeline(cfg=cfg, config_dir=config_dir, llm_fn=llm_fn)
    try:
        pipeline.run_dataset()
    except RuntimeError as exc:
        message = str(exc)
        if "prompt manifest only" in message:
            print(message)
            return
        raise


if __name__ == "__main__":
    main()
