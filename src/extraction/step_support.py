from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from extraction.backends.vllm_workers import GenerationTask
from extraction.evidence import build_scene_evidence
from extraction.progress import InferenceProgress
from extraction.scene_storage import ExtractionStepError, write_scene_records
from pipeline_runtime import RunContext
from preparation.manifest import selected_content_ids
from visual_sampling import timestamp_filename


def write_progress(progress: tqdm | InferenceProgress, message: str) -> None:
    if isinstance(progress, InferenceProgress):
        progress.write_log(message)
    else:
        tqdm.write(message, file=progress.fp)


def write_scene_results(scene_path: Path, records: list[dict[str, Any]]) -> None:
    """Publish completed scene results without a recovery journal."""
    records.sort(key=lambda row: int(row["scene_idx"]))
    write_scene_records(scene_path, records)


def scene_generation_rows(
    visual: dict[str, Any],
    *,
    prompt: str,
    max_new_tokens: int,
    generation: dict | None = None,
    scenes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if scenes is None:
        scenes = json.loads(Path(visual["timestamp_json"]).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for scene in build_scene_evidence(scenes, visual["frames_dir"]):
        scene_idx = scene["scene_idx"]
        keyframes = scene["keyframes"]
        image_paths = scene["image_paths"]
        if not keyframes or len(image_paths) != len(keyframes):
            raise ExtractionStepError(
                f"{visual['content_id']} scene {scene_idx} has "
                f"{len(image_paths)} of {len(keyframes)} keyframes"
            )
        task_id = f"{visual['content_id']}:{scene_idx}"
        rows.append(
            {
                "task": GenerationTask(
                    task_id=task_id,
                    image_paths=tuple(image_paths),
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    **(generation or {}),
                ),
                "scene_idx": scene_idx,
                "scene_start_seconds": scene["scene_start_seconds"],
                "scene_end_seconds": scene["scene_end_seconds"],
                "keyframes": keyframes,
                "image_paths": image_paths,
            }
        )
    if not rows:
        raise ExtractionStepError(f"{visual['content_id']} has no scenes")
    return rows


def result(
    stage: str,
    *,
    content_count: int,
    failure_count: int = 0,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "content_count": content_count,
        "failure_count": failure_count,
    }


def visual_rows(
    context: RunContext, *, split: str | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    """Prepared keyframe locations of the selected manifest videos."""
    sampling = context.config["extraction"]["visual_evidence"]
    name = timestamp_filename(sampling["scene_duration"], sampling["num_keyframes"])
    rows: list[dict[str, Any]] = []
    unprepared = 0
    for content_id in selected_content_ids(context, split=split):
        timestamp = context.source_assets_dir / content_id / name
        if not timestamp.is_file():
            # Not downloaded or not yet extracted; such videos never block the others.
            unprepared += 1
            continue
        if limit is not None and len(rows) >= limit:
            break
        rows.append(
            {
                "content_id": content_id,
                "frames_dir": str(context.keyframes_dir / content_id),
                "timestamp_json": str(timestamp),
            }
        )
    if unprepared:
        print(f"[PREPARE] skipping {unprepared} video(s) without prepared keyframes", flush=True)
    if not rows:
        raise ExtractionStepError(
            "no prepared videos selected; check the manifest, --split and --limit, and run "
            "python -m preparation extract-keyframes"
        )
    return rows
