"""Recall against a pooled reference.

No ground truth exists, so the reference of a scene is every entity, action and topic claim
that any compared model stated and the judge accepted. A model covers a reference claim it
stated itself; for the others the judge reads the model's graph as text and decides.
"""

from __future__ import annotations

from tqdm import tqdm

from artifact_io import atomic_write_json, atomic_write_jsonl, read_jsonl
from extraction.semantic_graph.serialize import serialize_graph
from extraction.step_support import visual_rows
from pipeline_logging import log_step_start
from scoring.d3_client import open_judge
from scoring.steps import scoring_settings, usable_scenes

REFERENCE_KINDS = ("entity", "action", "topic")
STATE_PREFIX = "This is a scene graph extracted from one 30-second video segment:\n\n"
COVERED = {
    "covered": "Yes, the graph contains it or a clear equivalent",
    "missing": "No, the graph does not contain it",
}


def reference_text(claim: dict) -> str:
    if claim["kind"] == "action":
        actor = (claim.get("actor") or "someone").split(" (")[0]
        return f"the action: {actor} {claim['value']}"
    return f"the {claim['kind']}: {claim['value']}"


def coverage(context, *, models, split=None, limit=None, threshold=None, d3_server=None):
    if len(models) < 2:
        raise ValueError("coverage compares at least two models")
    settings = scoring_settings(context)
    threshold = settings.get("reference_threshold", 0.7) if threshold is None else threshold
    output_root = context.run_root / "scoring" / "coverage"
    log_step_start(context, "coverage", models=models, threshold=threshold, output_dir=output_root,
                   **({"split": split} if split else {}), **({"limit": limit} if limit else {}))
    visuals = visual_rows(context, split=split, limit=limit)
    tasks, results = [], {model: {} for model in models}
    for visual in visuals:
        cid = visual["content_id"]
        graphs = {m: usable_scenes(context.scene_dir(m) / f"{cid}.jsonl")[0] for m in models}
        scored = {}
        for model in models:
            path = context.scoring_dir(model) / f"{cid}.jsonl"
            scored[model] = (
                {row["scene_idx"]: row for row in read_jsonl(path)} if path.is_file() else {}
            )
        for scene_idx in sorted(set().union(*(scored[m] for m in models))):
            reference = [
                (model, claim)
                for model in models
                for claim in scored[model].get(scene_idx, {}).get("claims", [])
                if claim["kind"] in REFERENCE_KINDS and (claim.get("effective") or 0) >= threshold
            ]
            if not reference:
                continue
            for model in models:
                row = {"content_id": cid, "scene_idx": scene_idx, "reference": len(reference),
                       "items": []}
                results[model].setdefault(cid, []).append(row)
                questions = {}
                for index, (source, claim) in enumerate(reference):
                    item = {"from": source, "kind": claim["kind"], "text": reference_text(claim)}
                    row["items"].append(item)
                    if source == model:
                        item["covered"] = 1.0
                    elif scene_idx not in graphs[model]:
                        item["covered"] = 0.0  # no usable graph for this scene
                    else:
                        questions[str(index)] = {
                            "type": "choice",
                            "instructions": f"Does the graph include {item['text']}? Count the "
                                            "same thing worded differently as included.",
                            "criteria": dict(COVERED),
                        }
                if questions:
                    tasks.append({
                        "task_id": f"{model}|{cid}|{scene_idx}",
                        "state": STATE_PREFIX + serialize_graph(graphs[model][scene_idx], "id"),
                        "questions": questions, "_row": row,
                    })
    owners = {task["task_id"]: task.pop("_row") for task in tasks}
    failures = 0
    with tqdm(total=len(tasks), desc="coverage", unit="scene", smoothing=0,
              dynamic_ncols=True) as bar:

        def receive(event):
            nonlocal failures
            row = owners[event["task_id"]]
            if "error" in event:
                failures += 1
                tqdm.write(f"[d3] {event['task_id']} failed: {event['error']}")
            else:
                for key, answer in event["answers"].items():
                    row["items"][int(key)]["covered"] = (
                        answer["probabilities"]["covered"] if "probabilities" in answer else None
                    )
            bar.update(1)

        if tasks:
            with open_judge(context, settings, server=d3_server, log=tqdm.write) as pool:
                pool.run(tasks, receive)
    summary = {"models": list(models), "threshold": threshold, "recall": {}}
    for model in models:
        values = []
        for cid, rows in results[model].items():
            for row in rows:
                known = [item["covered"] for item in row["items"] if item.get("covered") is not None]
                row["recall"] = sum(known) / len(known) if known else None
                if row["recall"] is not None:
                    values.append(row["recall"])
            atomic_write_jsonl(output_root / model / f"{cid}.jsonl", rows, durable=True,
                               sort_keys=False)
        summary["recall"][model] = {
            "scenes": len(values), "mean": sum(values) / len(values) if values else None,
        }
    atomic_write_json(output_root / "summary.json", summary, durable=True)
    print(f"[COVERAGE] {summary['recall']} failed={failures}", flush=True)
    return {"failure_count": failures}
