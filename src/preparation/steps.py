from __future__ import annotations

from pathlib import Path

from pipeline_logging import log_step_start
from preparation.download import download_settings, download_videos
from preparation.keyframes import extract_keyframes
from preparation.manifest import (
    MANIFEST_FIELDS, SPLIT_FIELDS, SPLITS, category_of, manifest_rows_from_txt,
    read_manifest_rows, read_splits, selected_content_ids, write_csv,
)
from preparation.split import assign_splits


def _preparation(context) -> dict:
    return context.config.get("preparation") or {}


def create_manifest(context, *, video_list=None, **_):
    """Write manifest.csv from the [Category] TXT list."""
    source = video_list or _preparation(context).get("video_list")
    if not source:
        raise ValueError("pass --list or set preparation.video_list")
    source = Path(source).expanduser()
    source = source if source.is_absolute() else context.root / source
    log_step_start(context, "create-manifest", video_list=source, output=context.manifest_path)
    rows = manifest_rows_from_txt(source)
    write_csv(context.manifest_path, MANIFEST_FIELDS, rows)
    print(f"[MANIFEST] videos={len(rows)} path={context.manifest_path}", flush=True)
    return {"failure_count": 0}


def make_split(context, *, force=False, **_):
    """Assign every manifest video to train/val/test; earlier assignments are kept."""
    settings = _preparation(context).get("split") or {}
    seed = settings.get("seed", 42)
    ratios = settings.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    log_step_start(context, "make-split", seed=seed, ratios=ratios, force=force,
                   output=context.splits_path)
    content_ids = [row["content_id"] for row in read_manifest_rows(context.manifest_path)]
    existing = {} if force or not context.splits_path.is_file() else read_splits(context.splits_path)
    assigned = assign_splits(content_ids, existing, seed=seed, ratios=ratios)
    write_csv(
        context.splits_path, SPLIT_FIELDS,
        [{"content_id": cid, "category": category_of(cid), "split": assigned[cid]}
         for cid in content_ids],
    )
    counts = {name: sum(split == name for split in assigned.values()) for name in SPLITS}
    kept = sum(cid in existing for cid in content_ids)
    print(f"[SPLIT] {counts} kept={kept} new={len(content_ids) - kept}", flush=True)
    return {"failure_count": 0}


def download(context, *, limit=None, content_ids=None, **_):
    settings = download_settings(context.config)
    shown = {key: value for key, value in settings.items() if key != "cookies_file"}
    log_step_start(context, "download-videos", data_root=context.data_root,
                   cookies=bool(settings["cookies_file"]), **shown)
    rows = read_manifest_rows(context.manifest_path)
    if content_ids:
        by_id = {row["content_id"]: row for row in rows}
        unknown = [cid for cid in content_ids if cid not in by_id]
        if unknown:
            raise ValueError(f"content ids not in the manifest: {', '.join(unknown)}")
        rows = [by_id[cid] for cid in content_ids]
    if limit is not None:
        rows = rows[:limit]
    result = download_videos(context, rows, settings=settings)
    print(f"[DOWNLOAD] present={result['present']} downloaded={result['downloaded']} "
          f"failed={result['failed']}", flush=True)
    return {"failure_count": result["failed"]}


def keyframes(context, *, force=False, limit=None, content_ids=None, **_):
    visual = context.config["extraction"]["visual_evidence"]
    workers = _preparation(context).get("keyframe_workers", 4)
    log_step_start(context, "extract-keyframes", output=context.keyframes_dir, force=force,
                   workers=workers, **visual)
    selected = content_ids or selected_content_ids(context, limit=limit)
    result = extract_keyframes(context, selected, force=force, workers=workers)
    print(f"[KEYFRAMES] videos={result['videos']} scenes={result['scenes']} "
          f"keyframes={result['keyframes']} new_frames={result['new_frames']} "
          f"not_downloaded={result['missing_video']} failed={result['failed']}", flush=True)
    return {"failure_count": result["failed"]}


STEP_HANDLERS = {
    "create-manifest": create_manifest,
    "make-split": make_split,
    "download-videos": download,
    "extract-keyframes": keyframes,
}
