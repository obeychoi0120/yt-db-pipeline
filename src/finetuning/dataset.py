"""Build the SFT dataset: keyframes + prompt -> the teacher's graph, split by video."""

from __future__ import annotations

from artifact_io import atomic_write_json, atomic_write_jsonl, file_fingerprint, read_json, read_jsonl
from extraction.semantic_graph import parse_or_repair_graph
from extraction.semantic_graph.serialize import prompt_form, serialize_graph
from extraction.step_support import visual_rows
from pipeline_logging import log_step_start
from scoring.steps import keyframe_paths, usable_scenes

DEFAULTS = {"min_scene_score": None, "score_units": ["context", "entity", "action"]}
IMAGE_TAG = "<image>"


def finetuning_settings(context) -> dict:
    return {**DEFAULTS, **(context.config.get("finetuning") or {})}


def scene_score(row: dict, units) -> float | None:
    """Mean of the selected unit scores of one scored scene."""
    values = [row["scores"][unit]["score"] for unit in units if unit in row["scores"]]
    return sum(values) / len(values) if values else None


def build_record(prompt: str, target: str, images: list[str], content_id: str, scene_idx: int):
    """One ms-swift record; the user turn is the images followed directly by the prompt,
    the same order the extraction backend renders."""
    return {
        "messages": [
            {"role": "user", "content": IMAGE_TAG * len(images) + prompt},
            {"role": "assistant", "content": target},
        ],
        "images": images,
        "content_id": content_id,
        "scene_idx": scene_idx,
    }


def build_dataset(context, *, teacher, schema, min_scene_score=None, allow_prompt_mismatch=False):
    settings = finetuning_settings(context)
    threshold = settings["min_scene_score"] if min_scene_score is None else min_scene_score
    prompt_path = context.prompt_path(schema)
    prompt = prompt_path.read_text(encoding="utf-8")
    form = prompt_form(prompt)
    output_dir = context.run_root / "sft"
    log_step_start(context, "build-dataset", teacher=teacher, prompt=prompt_path, form=form,
                   min_scene_score=threshold, score_units=settings["score_units"],
                   output_dir=output_dir)
    identity = read_json(context.scene_dir(teacher) / "generation_identity.json")["identity"]
    prompt_hash = file_fingerprint(prompt_path)
    if identity["prompt_hash"] != prompt_hash and not allow_prompt_mismatch:
        raise ValueError(
            f"the teacher graphs were generated with {identity['prompt_path']}, not {prompt_path}; "
            "training and inference must use the same prompt"
        )
    stats = {"teacher": teacher, "prompt_path": str(prompt_path), "prompt_hash": prompt_hash,
             "form": form, "min_scene_score": threshold, "score_units": settings["score_units"],
             "splits": {}}
    videos = {}
    for split in ("train", "val"):
        records = []
        counts = {"videos": 0, "scenes": 0, "kept": 0, "not_serializable": 0,
                  "below_score": 0, "not_scored": 0}
        try:
            visuals = visual_rows(context, split=split)
        except RuntimeError:
            visuals = []
        for visual in visuals:
            cid = visual["content_id"]
            graphs, _ = usable_scenes(context.scene_dir(teacher) / f"{cid}.jsonl")
            if not graphs:
                continue
            counts["videos"] += 1
            scores = {}
            score_path = context.scoring_dir(teacher) / f"{cid}.jsonl"
            if threshold is not None and score_path.is_file():
                scores = {row["scene_idx"]: row for row in read_jsonl(score_path)}
            frames = keyframe_paths(visual)
            for scene_idx, graph in sorted(graphs.items()):
                counts["scenes"] += 1
                if threshold is not None:
                    score = scene_score(scores[scene_idx], settings["score_units"]) \
                        if scene_idx in scores else None
                    if score is None:
                        counts["not_scored"] += 1
                        continue
                    if score < threshold:
                        counts["below_score"] += 1
                        continue
                try:
                    target = serialize_graph(graph, form)
                    # The target must read back as the same graph without repair.
                    parsed = parse_or_repair_graph(target)
                    if parsed.parse_mode != "native":
                        raise ValueError("target does not parse natively")
                except ValueError:
                    counts["not_serializable"] += 1
                    continue
                records.append(build_record(prompt, target, frames[scene_idx], cid, scene_idx))
                counts["kept"] += 1
        videos[split] = {record["content_id"] for record in records}
        atomic_write_jsonl(output_dir / f"{split}.jsonl", records, durable=True, sort_keys=False)
        stats["splits"][split] = counts
    shared = videos["train"] & videos["val"]
    if shared:
        raise RuntimeError(f"videos in both train and val: {sorted(shared)[:5]}")
    atomic_write_json(output_dir / "stats.json", stats, durable=True)
    print(f"[SFT] {stats['splits']} -> {output_dir}", flush=True)
    if not stats["splits"]["train"]["kept"]:
        raise RuntimeError("no training records; check the teacher extraction and the split")
    return {"failure_count": 0}
