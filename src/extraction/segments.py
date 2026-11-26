"""TV genre audience segment of each scene, chosen by the Gemini Decisions API.

Every segment of the table is one option of a single question, next to "none" for a scene that
fits no listed genre. The step shares the extraction.iab settings (API key, model, threads).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Sequence

from artifact_io import file_fingerprint
from extraction.iab import (
    MAX_OPTIONS, NONE, DecisionClient, drop_stale_labels, iab_settings, label_scenes,
    taxonomy_path,
)
from pipeline_logging import log_step_start

STAGE = "extract-segment-gt"
SEGMENT_VERSION = "segment-decision/v1"
INSTRUCTIONS = (
    "The images are keyframes sampled in order from one scene of a video. Choose the TV program "
    "genre that best describes what this scene is mainly about, as if the video were a program "
    "on linear TV. When a general genre and a more specific one both fit, choose the more "
    "specific one. Choose \"" + NONE + "\" only when no listed genre fits the scene."
)
NONE_DESCRIPTION = "No listed genre fits the scene."
# "Viewers in the top 20% of those who watch food or cooking contents on linear TV, ..."
WATCHED = re.compile(r"who watch (.+?) contents? on linear TV", re.IGNORECASE)


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def option_text(name: str, definition: str) -> str:
    """The segment name, plus what its audience watches when that says more than the name."""
    match = WATCHED.search(definition)
    if match is None or _key(match.group(1)) == _key(name):
        return name
    return f"{name} ({match.group(1).strip()})"


def load_segments(path: str | Path) -> dict[str, tuple[str, str]]:
    """Read the tab-separated segment table: option id -> (segment name, option text)."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    header = [cell.strip() for cell in rows[0]] if rows else []
    if "Segment Name" not in header or "Audience Definition" not in header:
        raise ValueError(
            f"{path}: no 'Segment Name' and 'Audience Definition' header columns (tab-separated)"
        )
    name_at, definition_at = header.index("Segment Name"), header.index("Audience Definition")
    segments: dict[str, tuple[str, str]] = {}
    for row in rows[1:]:
        cells = [cell.strip() for cell in row] + [""] * (len(header) - len(row))
        name = cells[name_at]
        if not name:
            continue
        option_id = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        if option_id in segments or option_id == NONE:
            raise ValueError(f"{path}: duplicate or reserved segment name {name!r}")
        segments[option_id] = (name, option_text(name, cells[definition_at]))
    if len(segments) < 2:
        raise ValueError(f"{path}: fewer than two segments")
    if len(segments) + 1 > MAX_OPTIONS:
        raise ValueError(f"{path}: {len(segments)} segments exceed the option limit")
    return segments


def classify_scene(
    client, images: Sequence[dict[str, str]], segments: dict[str, tuple[str, str]]
) -> dict:
    """One question over every segment; the fields of one scene's label."""
    options = {option_id: text for option_id, (_, text) in segments.items()}
    options[NONE] = NONE_DESCRIPTION
    choice, confidence, probabilities = client.choose(images, INSTRUCTIONS, options)
    chosen = None if choice == NONE else choice
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return {
        "segment_id": chosen,
        "segment": None if chosen is None else segments[chosen][0],
        "confidence": round(confidence, 4),
        "probabilities": {key: round(value, 4) for key, value in ranked},
    }


def extract_segment_gt(
    context, *, segments, split=None, limit=None, force=False, client_factory=DecisionClient,
):
    settings = iab_settings(context)
    path = taxonomy_path(context, segments, flag="--segments")
    output_dir = context.segment_dir
    log_step_start(
        context, STAGE, segments=path, model_id=settings["model_id"],
        threads=settings["threads"], output_dir=output_dir, force=force,
        **({"split": split} if split else {}),
        **({"limit": limit} if limit else {}),
    )
    table = load_segments(path)
    identity = {
        "version": SEGMENT_VERSION, "model_id": settings["model_id"],
        "segments": {"name": path.name, "sha256": file_fingerprint(path)},
    }
    drop_stale_labels(
        output_dir, identity, force=force, name="segment_identity.json", tag="SEGMENT",
        what="segment table",
    )
    return label_scenes(
        context, stage=STAGE, tag="SEGMENT", output_dir=output_dir, settings=settings,
        classify=lambda client, images: classify_scene(client, images, table),
        split=split, limit=limit, client_factory=client_factory,
    )
