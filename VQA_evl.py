from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from utils.evaluation_common import (
    DEFAULT_EVAL_CSV,
    DEFAULT_RESULTS_ROOT,
    discover_image_records,
    discover_method_dirs,
    ensure_evaluation_row,
    filter_image_records,
    load_evaluation_table,
    save_evaluation_table,
    sanitize_prompt_text,
    upsert_evaluation_row,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback only
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)


SYSTEM_PROMPT_TEMPLATE = (
    "You are a objective visual attribute evaluator. "
    "Observe the image and compare it against the description: '{prompt_text}'. "
    "Judge whether the primary entities are mostly bound with the correct main attributes, especially color and key appearance cues. "
    "Minor color mix, lighting variation, reflections, shading differences, or small background inconsistencies are acceptable, only if it is reasonable. "
    "Reply 'no' only when the main attributes are clearly wrong, swapped between entities, or seriously mixed up. "
    "Your reply must be exactly one lowercase word: yes or no. "
    "Do not output any punctuation, explanation, or extra words."
)


@dataclass
class VQARunConfig:
    results_root: str = str(DEFAULT_RESULTS_ROOT)
    csv_path: str = str(DEFAULT_EVAL_CSV)
    methods: List[str] = field(default_factory=list)
    scenes: List[str] = field(default_factory=list)
    prompt_ids: List[str] = field(default_factory=list)
    seeds: List[str] = field(default_factory=list)
    prompt_contains: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    overwrite: bool = False
    save_every: int = 1
    max_retries: int = 3
    retry_wait: float = 2.0
    max_workers: int = 5
    image_max_side: Optional[int] = 384
    limit: Optional[int] = None
    dry_run: bool = False
    show_llm_reply: bool = True
    verbose: bool = False


# Defaults for running directly from PyCharm.
# Use [] for methods/scenes/prompt_ids/seeds to include all records.
# For a quick smoke test, set dry_run=True or limit=1.
# If model is empty, src/llm_client.py resolves the default multimodal model.
PYCHARM_RUN_CONFIG = VQARunConfig(
    methods=[],
    scenes=[],
    prompt_ids=[],
    seeds=[],
    prompt_contains=None,
    provider="aixj_vip", # None
    model="gpt-5.4",
    max_workers=5,
    image_max_side=384,
    limit=None, # None
    dry_run=False,  # False
    show_llm_reply=True,
    overwrite=False,
    verbose=False,
)


def parse_args(run_config: Optional[VQARunConfig] = None) -> argparse.Namespace:
    cfg = run_config or PYCHARM_RUN_CONFIG
    parser = argparse.ArgumentParser(
        description="Run VQA-style attribute binding checks and update evaluation_results.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-root", default=cfg.results_root)
    parser.add_argument("--csv-path", default=cfg.csv_path)
    parser.add_argument("--methods", nargs="*", default=cfg.methods, help="Optional subset of method folder names.")
    parser.add_argument("--scenes", nargs="*", default=cfg.scenes, help="Optional exact scene-name filter.")
    parser.add_argument("--prompt-ids", nargs="*", default=cfg.prompt_ids, help="Optional exact prompt-folder filter, e.g. 1 2 5.")
    parser.add_argument("--seeds", nargs="*", default=cfg.seeds, help="Optional seed filter, e.g. 23 77 123.")
    parser.add_argument("--prompt-contains", default=cfg.prompt_contains, help="Optional substring filter on prompt text.")
    parser.add_argument("--provider", default=cfg.provider, help="LLM provider name in src.llm_client, e.g. dashscope or aixj.")
    parser.add_argument("--model", default=cfg.model, help="Override multimodal model name. Defaults to VLM_MODEL or qwen-vl-plus.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=cfg.overwrite)
    parser.add_argument("--save-every", type=int, default=cfg.save_every)
    parser.add_argument("--max-retries", type=int, default=cfg.max_retries)
    parser.add_argument("--retry-wait", type=float, default=cfg.retry_wait)
    parser.add_argument("--max-workers", type=int, default=cfg.max_workers, help="Maximum concurrent VQA requests. Will be clamped to 5.")
    parser.add_argument("--image-max-side", type=int, default=cfg.image_max_side, help="Resize the longer image side before upload. Set 0 to disable resizing.")
    parser.add_argument("--limit", type=int, default=cfg.limit, help="Only evaluate the first N discovered images.")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=cfg.dry_run, help="Only list matched records without calling the multimodal model.")
    parser.add_argument("--show-llm-reply", action=argparse.BooleanOptionalAction, default=cfg.show_llm_reply, help="Print raw LLM replies and parsed VQA results.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=cfg.verbose)
    return parser.parse_args()


def normalize_vqa_reply(text: str) -> Optional[int]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return None
    first = tokens[0]
    if first == "yes":
        return 1
    if first == "no":
        return 0

    token_set = set(tokens)
    if "yes" in token_set and "no" not in token_set:
        return 1
    if "no" in token_set and "yes" not in token_set:
        return 0
    return None


def should_skip_existing(existing_row: Optional[dict], overwrite: bool) -> bool:
    if overwrite or existing_row is None:
        return False
    return bool(str(existing_row.get("VQA_Result", "")).strip())


def get_multimodal_llm_api():
    try:
        from src.llm_client import call_multimodal_llm_api
        return call_multimodal_llm_api
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import `src.llm_client`. Please install the VQA dependencies first "
            "(for example: `pip install openai annotated-types tqdm`)."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize multimodal LLM client: {exc}") from exc


def build_system_prompt(prompt_text: str) -> str:
    description = " ".join(str(prompt_text or "").split()).strip()
    if not description:
        description = "[missing prompt text]"
    return SYSTEM_PROMPT_TEMPLATE.format(prompt_text=description)


def call_vqa_with_retry(
    *,
    prompt_text: str,
    image_path: Path,
    provider: Optional[str],
    model: Optional[str],
    max_retries: int,
    retry_wait: float,
    image_max_side: Optional[int],
) -> Tuple[int, str, int]:
    user_prompt = (
        f"Reference description: {prompt_text}\n"
        "Check whether the main entity-attribute bindings are overall successful, while tolerating minor visual variation."
    )
    multimodal_api = get_multimodal_llm_api()

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = multimodal_api(
                system_prompt=build_system_prompt(prompt_text),
                user_prompt=user_prompt,
                image_path=str(image_path),
                provider=provider,
                model=model,
                image_max_side=image_max_side,
            )
            parsed = normalize_vqa_reply(response)
            if parsed is not None:
                return parsed, str(response).strip(), attempt
            raise ValueError(f"Unexpected VQA reply: {response!r}")
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(max(0.0, retry_wait))

    if last_error is None:
        raise RuntimeError("VQA request failed without an explicit exception.")
    raise last_error


def preview_records(records, max_items: int = 10) -> None:
    print("[DRY RUN] Matched records only. No LLM request will be sent.")
    print(f"Matched records: {len(records)}")
    for record in records[:max(0, max_items)]:
        raw_prompt = record.prompt_text_raw or record.prompt_text
        sanitized_prompt = sanitize_prompt_text(raw_prompt)
        print(
            f"- method={record.method_name} | scene={record.scene_name} | "
            f"prompt_id={record.sample_id} | seed={record.seed_id} | image={record.image_path}"
        )
        if sanitized_prompt and sanitized_prompt != raw_prompt:
            print(f"  sanitized_prompt={sanitized_prompt}")
    if len(records) > max_items:
        print(f"... ({len(records) - max_items} more records omitted)")


def print_record_header(record, sanitized_prompt: str) -> None:
    print("=" * 90)
    print(f"Method     : {record.method_name}")
    print(f"Scene      : {record.scene_name}")
    print(f"Prompt ID  : {record.sample_id}")
    print(f"Seed       : {record.seed_id}")
    print(f"Image      : {record.image_path}")
    print(f"Prompt(raw): {record.prompt_text_raw or record.prompt_text}")
    print(f"Prompt(vqa): {sanitized_prompt}")


def print_run_mode(args, record_count: int) -> None:
    print("=" * 90)
    print("VQA evaluation started")
    print(f"Matched records : {record_count}")
    print(f"Dry run         : {args.dry_run}")
    print(f"Provider        : {args.provider or '[auto resolve from env / .codex / default]'}")
    print(f"Model override  : {args.model or '[default from llm_client / VLM_MODEL]'}")
    print(f"Max workers     : {max(1, min(5, int(args.max_workers or 1)))}")
    print(f"Image max side  : {args.image_max_side if (args.image_max_side or 0) > 0 else '[disabled]'}")
    print(f"CSV path        : {Path(args.csv_path).expanduser().resolve()}")
    print("=" * 90)


def main(run_config: Optional[VQARunConfig] = None) -> None:
    args = parse_args(run_config)
    results_root = Path(args.results_root).expanduser().resolve()
    csv_path = Path(args.csv_path).expanduser().resolve()

    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    method_dirs = discover_method_dirs(results_root, args.methods)
    if not method_dirs:
        raise RuntimeError(f"No evaluable method folders found under {results_root}")

    records = []
    for method_name, method_dir in method_dirs:
        records.extend(discover_image_records(method_name, method_dir))
    records = filter_image_records(
        records,
        scene_names=args.scenes,
        sample_ids=args.prompt_ids,
        seed_ids=args.seeds,
        prompt_contains=args.prompt_contains,
    )
    if args.limit is not None and args.limit >= 0:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("No generated images found under the selected results folders.")
    print_run_mode(args, len(records))
    if args.dry_run:
        preview_records(records)
        return

    table = load_evaluation_table(csv_path)
    stats = Counter()
    progress = tqdm(total=len(records), desc="VQA")
    max_workers = max(1, min(5, int(args.max_workers or 1)))

    interrupted = False
    try:
        pending_jobs = []
        completed_since_save = 0
        for record in records:
            existing = ensure_evaluation_row(
                table,
                method_name=record.method_name,
                scene_name=record.scene_name,
                prompt_text=record.prompt_text,
                seed_id=record.seed_id,
            )
            if should_skip_existing(existing, args.overwrite):
                stats["skipped_existing"] += 1
                progress.update(1)
                continue

            if not record.prompt_text:
                stats["missing_prompt"] += 1
                if args.verbose:
                    tqdm.write(f"[Skip] Missing prompt text for {record.image_path}")
                progress.update(1)
                continue

            sanitized_prompt = sanitize_prompt_text(record.prompt_text_raw or record.prompt_text)
            pending_jobs.append((record, sanitized_prompt))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(
                    call_vqa_with_retry,
                    prompt_text=sanitized_prompt,
                    image_path=record.image_path,
                    provider=args.provider,
                    model=args.model,
                    max_retries=max(1, args.max_retries),
                    retry_wait=args.retry_wait,
                    image_max_side=args.image_max_side,
                ): (record, sanitized_prompt)
                for record, sanitized_prompt in pending_jobs
            }

            for future in as_completed(future_to_job):
                record, sanitized_prompt = future_to_job[future]
                try:
                    result, raw_reply, used_attempt = future.result()
                    if args.show_llm_reply:
                        print_record_header(record, sanitized_prompt)
                        print("[Completed] Multimodal LLM request finished.")
                        print(f"[Reply] Attempt {used_attempt}: {raw_reply!r}")
                        print(f"[Parsed] VQA_Result = {result}")

                    upsert_evaluation_row(
                        table,
                        method_name=record.method_name,
                        scene_name=record.scene_name,
                        prompt_text=record.prompt_text,
                        seed_id=record.seed_id,
                        vqa_result=result,
                    )
                    if args.show_llm_reply:
                        print("[Saved] Result written to evaluation_results.csv")
                    stats["updated"] += 1
                except Exception as exc:
                    stats["errors"] += 1
                    tqdm.write(f"[Error] {record.image_path}: {exc}")
                    if args.verbose:
                        tqdm.write(traceback.format_exc())
                finally:
                    completed_since_save += 1
                    progress.update(1)
                    if completed_since_save % max(1, args.save_every) == 0:
                        save_evaluation_table(csv_path, table)
    except KeyboardInterrupt:
        interrupted = True
        tqdm.write("[Warn] Interrupted by user. Saving partial VQA results before exit...")
    finally:
        progress.close()

    save_evaluation_table(csv_path, table)

    print(f"Saved CSV: {csv_path}")
    print(
        "Summary:",
        {
            "updated": int(stats["updated"]),
            "skipped_existing": int(stats["skipped_existing"]),
            "missing_prompt": int(stats["missing_prompt"]),
            "errors": int(stats["errors"]),
            "records_total": len(records),
            "interrupted": interrupted,
        },
    )


if __name__ == "__main__":
    main()

