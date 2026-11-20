from __future__ import annotations

from pathlib import Path
from typing import Any
from visual_sampling import timestamp_stem, truncate_timestamp


def load_images(image_paths: list[str]) -> list[Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("image loading requires Pillow") from exc
    images: list[Any] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images


def build_scene_evidence(
    scenes: list[dict[str, Any]],
    frames_dir: str | Path,
) -> list[dict[str, Any]]:
    """Locate the prepared keyframes of each fixed window."""
    timeline = scenes
    rows: list[dict[str, Any]] = []
    for fallback_idx, scene in enumerate(scenes):
        keyframes = normalize_keyframe_timestamps(
            get_keyframe_timestamps(scene, timeline, fallback_idx)
        )
        rows.append(
            {
                "fallback_idx": fallback_idx,
                "scene_idx": int(scene.get("scene_idx", fallback_idx)),
                "scene_start_seconds": _scene_boundary(
                    scene, timeline, fallback_idx, "scene_start", keyframes[0] if keyframes else 0
                ),
                "scene_end_seconds": _scene_boundary(
                    scene,
                    timeline,
                    fallback_idx,
                    "scene_end",
                    keyframes[-1] if keyframes else 0,
                ),
                "keyframes": keyframes,
                "image_paths": [
                    str(Path(frames_dir) / f"{timestamp_stem(t)}.png") for t in keyframes
                ],
            }
        )
    return rows


def get_keyframe_timestamps(
    scene: dict[str, Any],
    timestamps: list[dict[str, Any]],
    fallback_idx: int,
) -> list[Any]:
    timeline = scene.get("timeline")
    if isinstance(timeline, list) and timeline:
        values = [
            shot.get("timestamp")
            for shot in timeline
            if isinstance(shot, dict) and shot.get("timestamp") is not None
        ]
        if len(values) == len(timeline):
            return values
    direct = scene.get("keyframe_timestamps") or scene.get("keyframes")
    if isinstance(direct, list) and direct:
        return direct
    index = scene.get("scene_idx", fallback_idx)
    if type(index) is not int:
        index = fallback_idx
    if timestamps and 0 <= index < len(timestamps):
        values = timestamps[index].get("keyframe_timestamps", [])
        if isinstance(values, list):
            return values
    return []


def normalize_keyframe_timestamps(values: list[Any]) -> list[int | float]:
    normalized: list[int | float] = []
    seen: set[int | float] = set()
    for value in values:
        try:
            timestamp = truncate_timestamp(value)
        except (TypeError, ValueError):
            continue
        if timestamp >= 0 and timestamp not in seen:
            normalized.append(timestamp)
            seen.add(timestamp)
    return sorted(normalized)


def _scene_boundary(
    scene: dict[str, Any],
    timeline: list[dict[str, Any]],
    fallback_idx: int,
    key: str,
    fallback: int | float,
) -> int | float:
    value = scene.get(key)
    if value is None and 0 <= fallback_idx < len(timeline):
        value = timeline[fallback_idx].get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return int(number) if number.is_integer() else number
