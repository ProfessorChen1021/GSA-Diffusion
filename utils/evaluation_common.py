from __future__ import annotations

import csv
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, MutableMapping, Optional, Sequence, Tuple


CSV_COLUMNS = [
    "Method_Name",
    "Scene_Name",
    "Prompt_Text",
    "Seed_ID",
    "R_CLIP",
    "L_CLIP",
    "VQA_Result",
    "Human_Score",
]

DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_MASK_ROOT = DEFAULT_RESULTS_ROOT / "mask"
DEFAULT_EVAL_CSV = DEFAULT_RESULTS_ROOT / "evaluation_results.csv"
IMAGE_FILENAMES = ("result.png", "generated.png")
TEXT_FALLBACK_FILENAMES = (
    "prompt_raw.txt",
    "original_prompt.txt",
    "base_prompt.txt",
    "prompt.txt",
    "prompt_used.txt",
    "regional_prompt.txt",
)
GENERATION_ONLY_SUFFIXES = {
    "photorealistic",
    "4k",
    "detailed reflections",
    "strong reflections",
    "highly detailed",
    "highly detailed fur",
    "cinematic lighting",
    "detailed textures",
    "studio lighting",
    "soft lighting",
    "soft warm lighting",
}


@dataclass(frozen=True)
class EntityInfo:
    name: str
    attr: str


@dataclass(frozen=True)
class MaskAssets:
    scene_name: str
    sample_id: str
    masks: Dict[str, Path]
    entities: List[EntityInfo]


@dataclass(frozen=True)
class EvaluationRecord:
    method_name: str
    scene_name: str
    sample_id: str
    prompt_text: str
    seed_id: str
    image_path: Path
    prompt_text_raw: str = ""


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def sanitize_prompt_text(prompt_text: str) -> str:
    raw_text = " ".join(str(prompt_text or "").split()).strip(" ,")
    if not raw_text:
        return ""

    kept_parts: List[str] = []
    for part in raw_text.split(","):
        cleaned_part = " ".join(part.split()).strip(" ,")
        if not cleaned_part:
            continue
        if cleaned_part.casefold() in GENERATION_ONLY_SUFFIXES:
            continue
        kept_parts.append(cleaned_part)

    sanitized = ", ".join(kept_parts).strip(" ,")
    return sanitized or raw_text


def _row_key(method_name: str, scene_name: str, prompt_text: str, seed_id: str) -> Tuple[str, str, str, str]:
    return (
        _safe_str(method_name),
        _safe_str(scene_name),
        sanitize_prompt_text(prompt_text),
        _safe_str(seed_id),
    )


def _new_row(method_name: str, scene_name: str, prompt_text: str, seed_id: str) -> Dict[str, str]:
    row = {col: "" for col in CSV_COLUMNS}
    row["Method_Name"] = _safe_str(method_name)
    row["Scene_Name"] = _safe_str(scene_name)
    row["Prompt_Text"] = sanitize_prompt_text(prompt_text)
    row["Seed_ID"] = _safe_str(seed_id)
    return row


def _sorted_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    def _seed_key(value: str) -> Tuple[int, str]:
        text = _safe_str(value)
        if text.isdigit():
            return (0, f"{int(text):09d}")
        return (1, text)

    return sorted(
        rows,
        key=lambda item: (
            _safe_str(item.get("Method_Name")),
            _safe_str(item.get("Scene_Name")),
            sanitize_prompt_text(_safe_str(item.get("Prompt_Text"))),
            _seed_key(_safe_str(item.get("Seed_ID"))),
        ),
    )


def load_json_file(path: Path) -> Optional[object]:
    if not os.path.exists(path):
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception:
            return None
    except Exception:
        return None


def read_text_file(path: Path) -> str:
    if not os.path.exists(path):
        return ""
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding).strip()
        except Exception:
            continue
    return ""


def parse_seed_id(text: str) -> str:
    candidate = _safe_str(text)
    if not candidate:
        return ""
    match = re.search(r"(\d+)", candidate)
    return match.group(1) if match else candidate


def discover_method_dirs(results_root: Path, methods: Optional[Sequence[str]] = None) -> List[Tuple[str, Path]]:
    root = results_root.expanduser().resolve()
    if not os.path.exists(root):
        raise FileNotFoundError(f"Results root does not exist: {root}")

    selected = {_safe_str(item) for item in (methods or []) if _safe_str(item)}
    parsed: List[Tuple[str, Path]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if child.name == "mask":
            continue
        if selected and child.name not in selected:
            continue
        parsed.append((child.name, child))
    return parsed


def build_mask_index(mask_root: Path) -> Dict[Tuple[str, str], MaskAssets]:
    root = mask_root.expanduser().resolve()
    if not os.path.exists(root):
        raise FileNotFoundError(f"Mask root does not exist: {root}")

    index: Dict[Tuple[str, str], MaskAssets] = {}
    for scene_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not scene_dir.is_dir():
            continue
        for sample_dir in sorted(scene_dir.iterdir(), key=lambda p: p.name):
            if not sample_dir.is_dir():
                continue
            processed_dir = sample_dir / "processed"
            entities_path = processed_dir / "entities.json"
            masks_dir = processed_dir / "masks"
            if not os.path.exists(entities_path) or not os.path.exists(masks_dir):
                continue

            entities_raw = load_json_file(entities_path)
            if not isinstance(entities_raw, list):
                continue

            mask_paths = {
                p.stem: p
                for p in masks_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            }
            if not mask_paths:
                continue

            entities: List[EntityInfo] = []
            for ent in entities_raw:
                if not isinstance(ent, dict):
                    continue
                name = _safe_str(ent.get("phrase_entity"))
                attr = _safe_str(ent.get("phrase_ent_attr")) or name
                if not name:
                    continue
                entities.append(EntityInfo(name=name, attr=attr))
            if not entities:
                continue

            key = (_safe_str(scene_dir.name), _safe_str(sample_dir.name))
            index[key] = MaskAssets(
                scene_name=scene_dir.name,
                sample_id=sample_dir.name,
                masks=mask_paths,
                entities=entities,
            )
    return index


def _json_pick(mapping: Optional[object], keys: Sequence[str]) -> str:
    if not isinstance(mapping, dict):
        return ""
    for key in keys:
        value = mapping.get(key)
        text = _safe_str(value)
        if text:
            return text
    return ""


def _collect_sidecars(seed_dir: Path, sample_dir: Path) -> Dict[str, object]:
    files = {
        "meta": seed_dir / "meta.json",
        "metadata": seed_dir / "metadata.json",
        "run_log": seed_dir / "run_log.json",
        "seed_summary": seed_dir / "seed_summary.json",
        "sample_summary": sample_dir / "sample_summary.json",
        "prompt_metadata": sample_dir / "prompt_metadata.json",
    }
    return {name: load_json_file(path) for name, path in files.items()}


def _extract_prompt_text(seed_dir: Path, sample_dir: Path, sidecars: Dict[str, object]) -> str:
    prompt = (
        _json_pick(sidecars.get("meta"), ("prompt_raw", "prompt", "prompt_used"))
        or _json_pick(sidecars.get("metadata"), ("prompt", "prompt_raw", "prompt_used"))
        or _json_pick(sidecars.get("run_log"), ("prompt", "prompt_raw", "prompt_used"))
        or _json_pick(sidecars.get("sample_summary"), ("input_prompt", "resolved_base_prompt", "base_prompt"))
    )
    if prompt:
        return prompt

    for filename in TEXT_FALLBACK_FILENAMES:
        text = read_text_file(sample_dir / filename)
        if text:
            return text

    planner = sidecars.get("sample_summary")
    if isinstance(planner, dict):
        planner_info = planner.get("planner")
        if isinstance(planner_info, dict):
            regional = _safe_str(planner_info.get("regional_prompt"))
            if regional:
                return regional
    return ""


def _extract_scene_name(sample_dir: Path, sidecars: Dict[str, object]) -> str:
    return (
        _safe_str(sample_dir.parent.name)
        or _json_pick(sidecars.get("meta"), ("dataset_name", "scene"))
        or _json_pick(sidecars.get("metadata"), ("scene", "dataset_name"))
        or _json_pick(sidecars.get("run_log"), ("scene", "dataset_name"))
        or _json_pick(sidecars.get("sample_summary"), ("scene", "dataset_name"))
        or _json_pick(sidecars.get("prompt_metadata"), ("scene", "dataset_name"))
    )


def _extract_sample_id(sample_dir: Path, sidecars: Dict[str, object]) -> str:
    return (
        _safe_str(sample_dir.name)
        or _json_pick(sidecars.get("meta"), ("sample_id", "case_id", "prompt_id"))
        or _json_pick(sidecars.get("metadata"), ("sample_id", "case_id", "prompt_id"))
        or _json_pick(sidecars.get("run_log"), ("sample_id", "case_id", "prompt_id"))
        or _json_pick(sidecars.get("sample_summary"), ("sample_id", "case_id", "prompt_id"))
        or _json_pick(sidecars.get("prompt_metadata"), ("sample_id", "case_id", "prompt_id"))
    )


def _extract_seed_id(seed_dir: Path, sidecars: Dict[str, object]) -> str:
    seed = (
        _json_pick(sidecars.get("meta"), ("seed",))
        or _json_pick(sidecars.get("metadata"), ("seed",))
        or _json_pick(sidecars.get("run_log"), ("seed",))
        or _json_pick(sidecars.get("seed_summary"), ("seed",))
    )
    return parse_seed_id(seed or seed_dir.name)


def discover_image_records(method_name: str, method_dir: Path) -> List[EvaluationRecord]:
    image_paths: List[Path] = []
    for filename in IMAGE_FILENAMES:
        image_paths.extend(method_dir.rglob(filename))

    dedup: "OrderedDict[Path, None]" = OrderedDict()
    for path in sorted(image_paths):
        dedup[path.resolve()] = None

    records: List[EvaluationRecord] = []
    for image_path in dedup.keys():
        try:
            seed_dir = image_path.parent
            sample_dir = seed_dir.parent
            if sample_dir == method_dir:
                continue
            sidecars = _collect_sidecars(seed_dir, sample_dir)
            scene_name = _extract_scene_name(sample_dir, sidecars)
            sample_id = _extract_sample_id(sample_dir, sidecars)
            prompt_text_raw = _extract_prompt_text(seed_dir, sample_dir, sidecars)
            prompt_text = sanitize_prompt_text(prompt_text_raw)
            seed_id = _extract_seed_id(seed_dir, sidecars)
            records.append(
                EvaluationRecord(
                    method_name=method_name,
                    scene_name=scene_name,
                    sample_id=sample_id,
                    prompt_text=prompt_text,
                    seed_id=seed_id,
                    image_path=image_path,
                    prompt_text_raw=prompt_text_raw,
                )
            )
        except Exception:
            continue
    return records


def _normalize_filter_value(value: object) -> str:
    return _safe_str(value).casefold()


def _build_filter_set(values: Optional[Sequence[str]]) -> set[str]:
    return {_normalize_filter_value(item) for item in (values or []) if _safe_str(item)}


def filter_image_records(
    records: Sequence[EvaluationRecord],
    *,
    scene_names: Optional[Sequence[str]] = None,
    sample_ids: Optional[Sequence[str]] = None,
    seed_ids: Optional[Sequence[str]] = None,
    prompt_contains: Optional[str] = None,
) -> List[EvaluationRecord]:
    scene_filter = _build_filter_set(scene_names)
    sample_filter = _build_filter_set(sample_ids)
    seed_filter = _build_filter_set(seed_ids)
    prompt_filter = _normalize_filter_value(prompt_contains) if _safe_str(prompt_contains) else ""

    filtered: List[EvaluationRecord] = []
    for record in records:
        if scene_filter and _normalize_filter_value(record.scene_name) not in scene_filter:
            continue
        if sample_filter and _normalize_filter_value(record.sample_id) not in sample_filter:
            continue
        if seed_filter and _normalize_filter_value(record.seed_id) not in seed_filter:
            continue
        if prompt_filter and prompt_filter not in _normalize_filter_value(record.prompt_text):
            continue
        filtered.append(record)
    return filtered


def load_evaluation_table(csv_path: Path) -> MutableMapping[Tuple[str, str, str, str], Dict[str, str]]:
    table: MutableMapping[Tuple[str, str, str, str], Dict[str, str]] = OrderedDict()
    if not os.path.exists(csv_path):
        return table

    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                if not raw:
                    continue
                row = {col: _safe_str(raw.get(col, "")) for col in CSV_COLUMNS}
                row["Prompt_Text"] = sanitize_prompt_text(row["Prompt_Text"])
                key = _row_key(
                    row["Method_Name"],
                    row["Scene_Name"],
                    row["Prompt_Text"],
                    row["Seed_ID"],
                )
                existing = table.get(key)
                if existing is None:
                    table[key] = row
                    continue
                for column in CSV_COLUMNS:
                    new_value = _safe_str(row.get(column, ""))
                    if new_value:
                        existing[column] = new_value
    except FileNotFoundError:
        return table
    return table


def save_evaluation_table(
    csv_path: Path,
    table: MutableMapping[Tuple[str, str, str, str], Dict[str, str]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in _sorted_rows(table.values()):
            serialized = {col: _safe_str(row.get(col, "")) for col in CSV_COLUMNS}
            serialized["Prompt_Text"] = sanitize_prompt_text(serialized["Prompt_Text"])
            writer.writerow(serialized)


def ensure_evaluation_row(
    table: MutableMapping[Tuple[str, str, str, str], Dict[str, str]],
    *,
    method_name: str,
    scene_name: str,
    prompt_text: str,
    seed_id: str,
) -> Dict[str, str]:
    key = _row_key(method_name, scene_name, prompt_text, seed_id)
    row = table.get(key)
    if row is None:
        row = _new_row(method_name, scene_name, prompt_text, seed_id)
        table[key] = row

    row["Method_Name"] = _safe_str(method_name)
    row["Scene_Name"] = _safe_str(scene_name)
    row["Prompt_Text"] = sanitize_prompt_text(prompt_text)
    row["Seed_ID"] = _safe_str(seed_id)
    if "Human_Score" not in row or row["Human_Score"] is None:
        row["Human_Score"] = ""
    return row


def upsert_evaluation_row(
    table: MutableMapping[Tuple[str, str, str, str], Dict[str, str]],
    *,
    method_name: str,
    scene_name: str,
    prompt_text: str,
    seed_id: str,
    r_clip: Optional[float] = None,
    l_clip: Optional[float] = None,
    vqa_result: Optional[int] = None,
) -> None:
    row = ensure_evaluation_row(
        table,
        method_name=method_name,
        scene_name=scene_name,
        prompt_text=prompt_text,
        seed_id=seed_id,
    )
    if r_clip is not None:
        row["R_CLIP"] = f"{float(r_clip):.6f}"
    if l_clip is not None:
        row["L_CLIP"] = f"{float(l_clip):.6f}"
    if vqa_result is not None:
        row["VQA_Result"] = str(int(vqa_result))


def iter_existing_records(
    table: MutableMapping[Tuple[str, str, str, str], Dict[str, str]]
) -> Iterator[Dict[str, str]]:
    for row in table.values():
        yield row

