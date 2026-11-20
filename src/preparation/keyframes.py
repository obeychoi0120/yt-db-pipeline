"""Fixed-window keyframes of every downloaded manifest video."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from artifact_io import write_jsonl
from preparation.video.fixed30 import prepare_visual_item, visual_evidence_matches
from preparation.video.media import resolve_duration
from visual_sampling import build_fixed_windows, timestamp_filename


def source_row(context, content_id: str) -> dict | None:
    """Identity of the downloaded video; None when it is not downloaded."""
    path = context.videos_dir / f"{content_id}.mp4"
    try:
        stat = path.stat()
    except OSError:
        return None
    return {
        "content_id": content_id,
        "source_video_path": str(path),
        "source_file_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def extract_keyframes(context, content_ids, *, force=False, workers=4) -> dict:
    settings = context.config["extraction"]["visual_evidence"]
    image_size = tuple(settings["image_resolution"])
    sampling = {key: settings[key] for key in ("scene_duration", "num_keyframes")}
    assets_root = context.source_assets_dir
    name = timestamp_filename(**sampling)

    def prepare(content_id):
        row = source_row(context, content_id)
        if row is None:
            return content_id, "missing_video", None
        try:
            duration = resolve_duration(assets_root, row)
            frames = context.keyframes_dir / content_id
            if not force and visual_evidence_matches(
                assets_root / content_id / name, frames, image_size, duration, **sampling
            ):
                counts = {"extracted_frames": 0}
            else:
                counts = prepare_visual_item(
                    content_id=content_id,
                    source_video_path=row["source_video_path"],
                    assets_root=assets_root,
                    output_root=context.preparation_dir,
                    duration_seconds=duration,
                    image_size=image_size,
                    force=force,
                    **sampling,
                )
            windows = build_fixed_windows(duration, **sampling)
            return content_id, None, {
                "scenes": len(windows),
                "keyframes": sum(len(w["keyframe_timestamps"]) for w in windows),
                "new_frames": counts["extracted_frames"],
            }
        except Exception as exc:  # one broken video must not stop the others
            return content_id, f"{type(exc).__name__}: {exc}", None

    totals = {"videos": 0, "scenes": 0, "keyframes": 0, "new_frames": 0, "missing_video": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for content_id, error, counts in pool.map(prepare, content_ids):
            if error == "missing_video":
                totals["missing_video"] += 1
            elif error is not None:
                failures.append({"content_id": content_id, "error": error})
                print(f"[KEYFRAMES] {content_id}: FAILED: {error}", flush=True)
            else:
                totals["videos"] += 1
                for key in ("scenes", "keyframes", "new_frames"):
                    totals[key] += counts[key]
    failure_path = context.preparation_dir / "keyframe_failures.jsonl"
    if failures:
        write_jsonl(failure_path, failures)
    else:
        failure_path.unlink(missing_ok=True)
    return {**totals, "failed": len(failures)}
