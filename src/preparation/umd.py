"""UMD category (subcategory) of each downloaded video, chosen by the Gemini Decisions API.

Up to max_frames keyframes, spread evenly over the whole video, are sent in one request with one
question whose options are every subcategory of the UMD class table, plus "none". The category
in the video's name is not trusted and is not given to the model; it is only recorded beside the
answer as query_category.
"""

from __future__ import annotations

import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

from artifact_io import atomic_write_jsonl, file_fingerprint, read_jsonl
from extraction.iab import (
    MAX_OPTIONS, NONE, DecisionClient, drop_stale_labels, iab_settings, image_blocks,
    keyframe_paths, taxonomy_path,
)
from extraction.progress import InferenceProgress
from extraction.step_support import write_progress
from pipeline_logging import log_step_start
from preparation.manifest import ManifestContractError, category_of, selected_content_ids, write_csv
from visual_sampling import timestamp_filename

STAGE = "classify-umd-categories"
UMD_VERSION = "umd-decision/v1"
DEFAULTS = {
    "classes": None, "max_frames": 50, "low_confidence": 0.3, "threads": 8,
}
TOP_CATEGORIES = 5
LOW_CONFIDENCE = "LOW_CONFIDENCE"
OUTCOMES = ("ok", "failed")
CLASS_FIELDS = ("Category", "Subcategory Count", "Subcategories")
CSV_FIELDS = (
    "content_id", "query_category", "pred_umd_categories", "warning", "frames_total",
    "frames_used", "error",
)
INSTRUCTIONS = (
    "The images are keyframes sampled in order across one whole video. Choose the genre or "
    "topic that best describes what this video is mainly about. When a general option and a "
    "more specific one both fit, choose the more specific one. Choose \"" + NONE + "\" only "
    "when no listed option fits the video."
)
NONE_DESCRIPTION = "No listed genre or topic fits the video."


def load_classes(path: str | Path) -> dict[str, str]:
    """Read the UMD class table: subcategory -> its category, in table order."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or tuple(cell.strip() for cell in rows[0][:3]) != CLASS_FIELDS:
        raise ValueError(f"{path}: the header must start with {', '.join(CLASS_FIELDS)}")
    classes: dict[str, str] = {}
    for row in rows[1:]:
        cells = [cell.strip() for cell in row[:3]] + [""] * (3 - len(row[:3]))
        category, count, names = cells
        if not category:
            continue
        subcategories = [name.strip() for name in names.split(",") if name.strip()]
        if not count.isdigit() or int(count) != len(subcategories):
            raise ValueError(
                f"{path}: {category} lists {len(subcategories)} subcategories, "
                f"but Subcategory Count is {count!r}"
            )
        for name in subcategories:
            if name in classes or name.lower() == NONE:
                raise ValueError(f"{path}: duplicate or reserved subcategory {name!r}")
            classes[name] = category
    if len(classes) < 2:
        raise ValueError(f"{path}: fewer than two subcategories")
    if len(classes) + 1 > MAX_OPTIONS:
        raise ValueError(f"{path}: {len(classes)} subcategories exceed the option limit")
    return classes


def options(classes: dict[str, str]) -> dict[str, str]:
    """Option id is the subcategory name; its text names the category, e.g. Law (Action and
    Thriller), so that short names such as Law, Game or Go are not misread."""
    texts = {name: f"{name} ({category.replace('_', ' ')})" for name, category in classes.items()}
    texts[NONE] = NONE_DESCRIPTION
    return texts


def select_frames(paths: Sequence[str], max_frames: int) -> list[str]:
    """All frames when they fit; otherwise max_frames evenly spaced, first and last included."""
    if len(paths) <= max_frames:
        return list(paths)
    if max_frames == 1:
        return [paths[0]]
    last = len(paths) - 1
    return [paths[round(index * last / (max_frames - 1))] for index in range(max_frames)]


def warning(pred_umd_categories: dict[str, float] | None, low_confidence: float) -> str | None:
    if not pred_umd_categories:
        return None
    return LOW_CONFIDENCE if max(pred_umd_categories.values()) < low_confidence else None


def classify_video(client, images, choices: dict[str, str]) -> dict[str, float]:
    """The TOP_CATEGORIES most probable subcategories ("none" included) of one video."""
    _, _, probabilities = client.choose(images, INSTRUCTIONS, choices)
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return {name: round(value, 4) for name, value in ranked[:TOP_CATEGORIES]}


def umd_settings(context) -> dict:
    settings = {**DEFAULTS, **((context.config.get("preparation") or {}).get("umd") or {})}
    for key in ("max_frames", "threads"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"preparation.umd.{key} must be a positive integer")
    threshold = settings["low_confidence"]
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
        raise ValueError("preparation.umd.low_confidence must be a number from 0 to 1")
    return settings


def query_category(content_id: str) -> str | None:
    try:
        return category_of(content_id)
    except ManifestContractError:
        return None


def prepared_videos(context, *, content_ids=None, limit=None) -> list[dict[str, str]]:
    """Keyframe locations of the selected manifest videos whose keyframes are extracted."""
    sampling = context.config["extraction"]["visual_evidence"]
    name = timestamp_filename(sampling["scene_duration"], sampling["num_keyframes"])
    rows, unprepared = [], 0
    for content_id in content_ids or selected_content_ids(context):
        timestamp = context.source_assets_dir / content_id / name
        if not timestamp.is_file():
            unprepared += 1
            continue
        if limit is not None and len(rows) >= limit:
            break
        rows.append({
            "content_id": content_id,
            "frames_dir": str(context.keyframes_dir / content_id),
            "timestamp_json": str(timestamp),
        })
    if unprepared:
        print(f"[UMD] skipping {unprepared} video(s) without extracted keyframes", flush=True)
    if not rows:
        raise ValueError(
            "no video with extracted keyframes selected; run python -m preparation "
            "extract-keyframes first"
        )
    return rows


def _csv_row(row: dict) -> dict:
    predictions = row.get("pred_umd_categories") or {}
    return {
        **{field: row.get(field) for field in CSV_FIELDS},
        "pred_umd_categories": "; ".join(f"{name}:{value}" for name, value in predictions.items()),
        "warning": row.get("warning") or "",
        "error": row.get("error") or "",
    }


def classify_umd_categories(
    context, *, classes=None, content_ids=None, limit=None, force=False,
    client_factory=DecisionClient,
):
    decision = iab_settings(context)
    settings = umd_settings(context)
    source = classes or settings["classes"]
    if not source:
        raise ValueError("pass --classes or set preparation.umd.classes")
    path = taxonomy_path(context, source, flag="--classes")
    output_dir = context.umd_dir
    labels_path = output_dir / "labels.jsonl"
    log_step_start(
        context, STAGE, classes=path, model_id=decision["model_id"],
        max_frames=settings["max_frames"], low_confidence=settings["low_confidence"],
        threads=settings["threads"],
        output_dir=output_dir, force=force,
        **({"limit": limit} if limit else {}),
    )
    choices = options(load_classes(path))
    identity = {
        "version": UMD_VERSION, "model_id": decision["model_id"],
        "classes": {"name": path.name, "sha256": file_fingerprint(path)},
        "max_frames": settings["max_frames"],
    }
    drop_stale_labels(
        output_dir, identity, force=force, name="umd_identity.json", tag="UMD",
        what="class table, frame limit",
    )
    visuals = prepared_videos(context, content_ids=content_ids, limit=limit)
    # Labels of videos outside this selection are kept.
    rows = {row["content_id"]: row for row in read_jsonl(labels_path)} if (
        labels_path.is_file()
    ) else {}
    pending = [visual for visual in visuals
               if visual["content_id"] not in rows or "error" in rows[visual["content_id"]]]
    reused = len(visuals) - len(pending)
    client = client_factory(api_key=decision["api_key"].strip(), model_id=decision["model_id"])
    lock = threading.Lock()

    def save() -> None:
        atomic_write_jsonl(
            labels_path, [rows[cid] for cid in sorted(rows)], durable=True, sort_keys=False,
        )

    def label(visual: dict) -> str:
        cid = visual["content_id"]
        row = {"content_id": cid, "query_category": query_category(cid),
               "pred_umd_categories": None, "warning": None}
        try:
            frames = [frame for scene in keyframe_paths(visual).values() for frame in scene]
            used = select_frames(frames, settings["max_frames"])
            row.update(frames_total=len(frames), frames_used=len(used))
            row["pred_umd_categories"] = classify_video(
                client, image_blocks(used), choices
            )
        except Exception as exc:  # one failed video never stops the others
            row["error"] = f"{type(exc).__name__}: {exc}"
        with lock:
            rows[cid] = row
            save()
        return "failed" if "error" in row else "ok"

    with InferenceProgress(
        total=len(pending), reused=reused, desc=output_dir.name, unit="video", outcomes=OUTCOMES,
    ) as progress:
        executor = ThreadPoolExecutor(settings["threads"], thread_name_prefix="umd")
        try:
            futures = {executor.submit(label, visual): visual["content_id"] for visual in pending}
            for future in as_completed(futures):
                cid = futures[future]
                outcome = future.result()
                progress.complete(outcome=outcome, task_id=cid)
                if outcome == "failed":
                    write_progress(progress, f"[UMD] {cid}: {rows[cid]['error']}")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    # The warning follows the current threshold, also for labels made by an earlier run.
    for row in rows.values():
        row["warning"] = warning(row.get("pred_umd_categories"), settings["low_confidence"])
    save()
    write_csv(output_dir / "labels.csv", CSV_FIELDS,
              [_csv_row(rows[cid]) for cid in sorted(rows)])
    tokens = getattr(client, "input_tokens", 0)
    if tokens:
        print(f"[UMD] input_tokens={tokens} over {len(pending)} video(s)", flush=True)
    selected = [rows[visual["content_id"]] for visual in visuals]
    failures = sum("error" in row for row in selected)
    low = sum(row["warning"] == LOW_CONFIDENCE for row in selected)
    print(f"[UMD] videos={len(selected)} low_confidence={low} failed={failures} "
          f"path={labels_path}", flush=True)
    return {"failure_count": failures}
