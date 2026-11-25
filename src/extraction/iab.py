"""IAB Content Taxonomy labels of each scene, chosen by the Gemini Decisions API.

A scene's keyframes are classified one tier at a time: first among the Tier 1 categories, then
among the children of the chosen category, until a leaf is reached or the model answers that no
child fits. Every question stays far below the API's 255-option limit.
"""

from __future__ import annotations

import base64
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from artifact_io import (
    atomic_write_json, atomic_write_jsonl, file_fingerprint, read_json, read_jsonl,
)
from extraction.backends.gemini import (
    RESOURCE_EXHAUSTED_MAX_ATTEMPTS, RESOURCE_EXHAUSTED_RETRY_DELAY_SECONDS, _google_genai,
    _is_resource_exhausted,
)
from extraction.progress import InferenceProgress
from extraction.step_support import result, visual_rows, write_progress
from pipeline_logging import log_step_start
from visual_sampling import timestamp_stem

STAGE = "extract-iab-gt"
IAB_VERSION = "iab-decision/v1"
DEFAULTS = {"model_id": "gemini-decision-1", "threads": 16}
# The Decisions endpoint is served on v1beta only; the SDK default (v1) answers 404.
API_VERSION = "v1beta"
OUTCOMES = ("ok", "failed")
NONE = "none"
MAX_OPTIONS = 255
TOP_PROBABILITIES = 3
ROOT_INSTRUCTIONS = (
    "The images are keyframes sampled in order from one scene of a video. Choose the IAB "
    "Content Taxonomy category that best describes what this scene is mainly about."
)
CHILD_INSTRUCTIONS = (
    "The images are keyframes sampled in order from one scene of a video. The scene is about "
    "\"{path}\". Choose the subcategory that best describes it. Choose \"" + NONE + "\" only "
    "when no listed subcategory fits the scene."
)
NONE_DESCRIPTION = "No listed subcategory fits; the scene is only generally about {name}."


@dataclass(frozen=True)
class Taxonomy:
    names: dict[str, str]
    parents: dict[str, str | None]
    children: dict[str | None, tuple[str, ...]]

    def path(self, node_id: str) -> list[str]:
        names = []
        current: str | None = node_id
        while current is not None:
            names.append(self.names[current])
            current = self.parents[current]
        return names[::-1]


def load_taxonomy(path: str | Path) -> Taxonomy:
    """Read the tab-separated IAB table: Unique ID, Parent, Name after a group header row."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    start = next(
        (index for index, row in enumerate(rows) if row and row[0].strip() == "Unique ID"), None
    )
    if start is None or [cell.strip() for cell in rows[start][:3]] != [
        "Unique ID", "Parent", "Name"
    ]:
        raise ValueError(f"{path}: no 'Unique ID / Parent / Name' header row (tab-separated)")
    names: dict[str, str] = {}
    parents: dict[str, str | None] = {}
    children: dict[str | None, list[str]] = {}
    for row in rows[start + 1:]:
        cells = [cell.strip() for cell in row[:3]] + [""] * (3 - len(row[:3]))
        node_id, parent, name = cells
        if not node_id:
            continue
        if not name:
            raise ValueError(f"{path}: category {node_id} has no name")
        if node_id in names or node_id == NONE:
            raise ValueError(f"{path}: duplicate or reserved category id {node_id}")
        names[node_id] = name
        parents[node_id] = parent or None
        children.setdefault(parent or None, []).append(node_id)
    for parent, ids in children.items():
        if parent is not None and parent not in names:
            raise ValueError(f"{path}: category {ids[0]} names a missing parent {parent}")
        if len(ids) + 1 > MAX_OPTIONS:
            raise ValueError(f"{path}: {len(ids)} categories under one parent exceed the limit")
    if len(children.get(None, ())) < 2:
        raise ValueError(f"{path}: fewer than two top-level categories")
    return Taxonomy(names, parents, {key: tuple(ids) for key, ids in children.items()})


class DecisionClient:
    """One single-select question per call against the Gemini Developer API."""

    def __init__(self, *, api_key: str, model_id: str, client: Any = None) -> None:
        if client is None:
            genai, types = _google_genai()
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=300_000, retry_options=types.HttpRetryOptions(attempts=1)
                ),
            )
            if not hasattr(client, "decisions"):
                raise RuntimeError(
                    "this google-genai has no Decisions API; pip install -U 'google-genai>=3.6'"
                )
        self.client = client
        self.model_id = model_id
        self.input_tokens = 0
        self._lock = threading.Lock()

    def choose(
        self, images: Sequence[dict[str, str]], instructions: str, options: dict[str, str]
    ) -> tuple[str, float, dict[str, float]]:
        questions = {"q": {"type": "choice", "instructions": instructions, "criteria": options}}
        for attempt in range(RESOURCE_EXHAUSTED_MAX_ATTEMPTS):
            try:
                response = self.client.decisions.create(
                    model=self.model_id, input=list(images), questions=questions,
                    api_version=API_VERSION,
                )
                break
            except Exception as exc:
                if attempt + 1 == RESOURCE_EXHAUSTED_MAX_ATTEMPTS or not _is_resource_exhausted(
                    exc
                ):
                    raise
                time.sleep(RESOURCE_EXHAUSTED_RETRY_DELAY_SECONDS)
        tokens = getattr(getattr(response, "usage", None), "total_input_tokens", None)
        if tokens:
            with self._lock:
                self.input_tokens += tokens
        answer = (getattr(response, "answers", None) or {}).get("q")
        choice = getattr(answer, "choice", None)
        if choice not in options:
            feedback = getattr(response, "prompt_feedback", None)
            reason = getattr(feedback, "block_reason", None)
            raise RuntimeError(
                f"no usable decision answer (choice={choice!r}, block_reason={reason})"
            )
        return choice, float(answer.confidence), dict(answer.probabilities)


def image_blocks(image_paths: Sequence[str | Path]) -> list[dict[str, str]]:
    return [
        {
            "type": "image",
            "mime_type": "image/png",
            "data": base64.b64encode(Path(path).read_bytes()).decode("ascii"),
        }
        for path in image_paths
    ]


def _top(probabilities: dict[str, float]) -> dict[str, float]:
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return {key: round(value, 4) for key, value in ranked[:TOP_PROBABILITIES]}


def classify_scene(client, images: Sequence[dict[str, str]], taxonomy: Taxonomy) -> dict:
    """Descend from Tier 1 until a leaf or a "none" answer; the fields of one scene's label."""
    tiers: list[dict[str, Any]] = []
    stop_confidence = None
    current: str | None = None
    while taxonomy.children.get(current):
        options = {child: taxonomy.names[child] for child in taxonomy.children[current]}
        if current is None:
            instructions = ROOT_INSTRUCTIONS
        else:
            instructions = CHILD_INSTRUCTIONS.format(path=" > ".join(taxonomy.path(current)))
            options[NONE] = NONE_DESCRIPTION.format(name=taxonomy.names[current])
        choice, confidence, probabilities = client.choose(images, instructions, options)
        if choice == NONE:
            stop_confidence = round(confidence, 4)
            break
        tiers.append(
            {
                "tier": len(tiers) + 1, "id": choice, "name": taxonomy.names[choice],
                "confidence": round(confidence, 4), "top": _top(probabilities),
            }
        )
        current = choice
    label = {
        "iab_id": current, "iab_name": taxonomy.names[current],
        "iab_path": taxonomy.path(current), "confidence": tiers[-1]["confidence"],
        "tiers": tiers,
    }
    if stop_confidence is not None:
        label["stop_confidence"] = stop_confidence
    return label


def iab_settings(context) -> dict:
    settings = {**DEFAULTS, **(context.config["extraction"].get("iab") or {})}
    key = settings.get("api_key")
    if not isinstance(key, str) or not key.strip() or key.strip().startswith("YOUR_"):
        raise ValueError("extraction.iab.api_key is missing; set a Gemini API key in config.yaml")
    if not isinstance(settings["threads"], int) or settings["threads"] <= 0:
        raise ValueError("extraction.iab.threads must be a positive integer")
    return settings


def taxonomy_path(context, taxonomy: str | Path) -> Path:
    path = Path(taxonomy).expanduser()
    if not path.is_absolute():
        path = context.root / path
    if not path.is_file():
        raise ValueError(f"--taxonomy file not found: {path}")
    return path


def keyframe_paths(visual: dict) -> dict[int, list[str]]:
    windows = json.loads(Path(visual["timestamp_json"]).read_text(encoding="utf-8"))
    return {
        index: [str(Path(visual["frames_dir"]) / f"{timestamp_stem(t)}.png")
                for t in window["keyframe_timestamps"]]
        for index, window in enumerate(windows)
    }


def _drop_stale_labels(output_dir: Path, identity: dict, *, force: bool) -> None:
    """Labels made with another taxonomy, model or question wording are deleted first."""
    path = output_dir / "iab_identity.json"
    saved = read_json(path) if path.is_file() else None
    stale = sorted(output_dir.glob("*.jsonl")) if output_dir.is_dir() else []
    if stale and (force or saved != identity):
        why = "--force" if force else "taxonomy or model changed"
        print(f"[IAB] {why}; removing {len(stale)} label files in {output_dir}", flush=True)
        for old in stale:
            old.unlink()
    atomic_write_json(path, identity, durable=True)


def extract_iab_gt(
    context, *, taxonomy, split=None, limit=None, force=False, client_factory=DecisionClient,
):
    settings = iab_settings(context)
    path = taxonomy_path(context, taxonomy)
    output_dir = context.iab_dir
    log_step_start(
        context, STAGE, taxonomy=path, model_id=settings["model_id"],
        threads=settings["threads"], output_dir=output_dir, force=force,
        **({"split": split} if split else {}),
        **({"limit": limit} if limit else {}),
    )
    tree = load_taxonomy(path)
    identity = {
        "version": IAB_VERSION, "model_id": settings["model_id"],
        "taxonomy": {"name": path.name, "sha256": file_fingerprint(path)},
    }
    _drop_stale_labels(output_dir, identity, force=force)
    visuals = visual_rows(context, split=split, limit=limit)
    rows: dict[str, dict[int, dict]] = {}
    pending: list[tuple[str, int, list[str]]] = []
    reused = 0
    for visual in visuals:
        cid = visual["content_id"]
        label_path = output_dir / f"{cid}.jsonl"
        cached = {row["scene_idx"]: row for row in read_jsonl(label_path)} if (
            label_path.is_file()
        ) else {}
        rows[cid] = {}
        for scene_idx, images in keyframe_paths(visual).items():
            row = cached.get(scene_idx)
            if row is not None and "error" not in row:
                rows[cid][scene_idx] = row
                reused += 1
            else:
                pending.append((cid, scene_idx, images))
    client = client_factory(api_key=settings["api_key"].strip(), model_id=settings["model_id"])
    locks = {cid: threading.Lock() for cid in rows}

    def label(cid: str, scene_idx: int, images: list[str]) -> str:
        row = {"content_id": cid, "scene_idx": scene_idx}
        try:
            row.update(classify_scene(client, image_blocks(images), tree))
        except Exception as exc:  # one failed scene never stops the others
            row["error"] = f"{type(exc).__name__}: {exc}"
        with locks[cid]:
            rows[cid][scene_idx] = row
            atomic_write_jsonl(
                output_dir / f"{cid}.jsonl",
                [rows[cid][index] for index in sorted(rows[cid])],
                durable=True, sort_keys=False,
            )
        return "failed" if "error" in row else "ok"

    with InferenceProgress(
        total=len(pending), reused=reused, desc="iab_gt", unit="scene", outcomes=OUTCOMES,
    ) as progress:
        executor = ThreadPoolExecutor(settings["threads"], thread_name_prefix="iab")
        try:
            futures = {
                executor.submit(label, cid, scene_idx, images): (cid, scene_idx)
                for cid, scene_idx, images in pending
            }
            for future in as_completed(futures):
                cid, scene_idx = futures[future]
                outcome = future.result()
                progress.complete(outcome=outcome, task_id=f"{cid}:{scene_idx}")
                if outcome == "failed":
                    write_progress(
                        progress, f"[IAB] {cid} scene {scene_idx}: {rows[cid][scene_idx]['error']}"
                    )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    tokens = getattr(client, "input_tokens", 0)
    if tokens:
        print(f"[IAB] input_tokens={tokens} over {len(pending)} scene(s)", flush=True)
    failure_count = sum("error" in row for scenes in rows.values() for row in scenes.values())
    return result(STAGE, content_count=len(visuals), failure_count=failure_count)
