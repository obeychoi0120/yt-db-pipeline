from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from artifact_io import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl
from extraction.graph_warnings import blocking_tags
from extraction.step_support import visual_rows
from pipeline_logging import log_step_start
from pipeline_runtime import scene_arm
from scoring.claims import foil_claims, graph_claims
from scoring.d3_client import judge_name, open_judge
from scoring.questions import STATE, claim_score, question, unit_scores
from visual_sampling import timestamp_stem

SCORING_VERSION = "d3-claims/v2"
DEFAULTS = {
    "batch_size": 8, "foil_seed": 42, "foil_entities": 1, "foil_actions": 0,
    # --d3-server only: the model name sent to the server and the requests kept in flight
    "server_model": "vllm-sr/d3-mini", "server_concurrency": 32,
}


def scoring_settings(context) -> dict:
    return {**DEFAULTS, **(context.config.get("scoring") or {})}


def usable_scenes(path: Path) -> tuple[dict[int, dict], int]:
    """scene_idx -> parsed graph of the usable rows, and the number of stored rows."""
    if not path.is_file():
        return {}, 0
    rows = read_jsonl(path)
    graphs = {
        row["scene_idx"]: row["scene_graph"]
        for row in rows
        if "error" not in row and isinstance(row.get("scene_graph"), dict)
        and not blocking_tags(row.get("warning"))
    }
    return graphs, len(rows)


def keyframe_paths(visual: dict) -> dict[int, list[str]]:
    windows = json.loads(Path(visual["timestamp_json"]).read_text(encoding="utf-8"))
    return {
        index: [str(Path(visual["frames_dir"]) / f"{timestamp_stem(t)}.png")
                for t in window["keyframe_timestamps"]]
        for index, window in enumerate(windows)
    }


CLAIM_FIELDS = ("id", "unit", "kind", "value", "attributes", "actor", "target", "tool", "place",
                "parent")
SCORE_FIELDS = ("p", "abstain", "judge_choice", "effective", "note")
IDENTITY_KEYS = ("version", "judge", "state", "foils")


def _stored_claim(claim: dict) -> dict:
    return {key: claim[key] for key in (*CLAIM_FIELDS, *SCORE_FIELDS) if key in claim}


def _claim_keys(claims) -> list[dict]:
    """The claims without their scores: equal when the scene graph is unchanged."""
    return [{key: claim[key] for key in CLAIM_FIELDS if key in claim} for claim in claims]


def _drop_stale_scores(output_dir, identity: dict) -> None:
    """Scores made with other settings (judge, state, foils) are deleted before scoring."""
    path = output_dir / "scoring_identity.json"
    if not path.is_file():
        return
    saved = read_json(path)
    if all(saved.get(key) == identity[key] for key in IDENTITY_KEYS):
        return
    stale = sorted(output_dir.glob("*.jsonl"))
    print(f"[SCORE] scoring settings changed; removing {len(stale)} score files in {output_dir}",
          flush=True)
    for old in stale:
        old.unlink()
    path.unlink()


def score_graphs(context, *, model, split=None, limit=None, force=False, d3_server=None):
    settings = scoring_settings(context)
    judge = judge_name(context, settings, server=d3_server)
    output_dir = context.scoring_dir(model)
    log_step_start(context, "score", model=model, d3=d3_server or judge, output_dir=output_dir,
                   force=force, **({"split": split} if split else {}),
                   **({"limit": limit} if limit else {}), **settings)
    scene_dir = context.scene_dir(model)
    visuals = visual_rows(context, split=split, limit=limit)
    graphs, stored_rows = {}, 0
    for visual in visuals:
        graphs[visual["content_id"]], count = usable_scenes(
            scene_dir / f"{visual['content_id']}.jsonl"
        )
        stored_rows += count
    if not stored_rows:
        raise RuntimeError(f"no extracted scenes for {model}: {scene_dir}")
    identity = {
        "version": SCORING_VERSION, "judge": judge, "state": STATE,
        "foils": {key: settings[key] for key in ("foil_seed", "foil_entities", "foil_actions")},
    }
    _drop_stale_scores(output_dir, identity)
    # Names and phrases of every other video are the foil pool of one video.
    vocabulary = {
        cid: {
            "entities": {e["name"] for g in scenes.values() for e in g["entities"]},
            "actions": {a["action"] for g in scenes.values() for a in g["actions"]},
        }
        for cid, scenes in graphs.items()
    }
    rows, pending, tasks = {}, {}, []
    for visual in visuals:
        cid = visual["content_id"]
        path = output_dir / f"{cid}.jsonl"
        cached = {}
        if path.is_file() and not force:
            cached = {row["scene_idx"]: row for row in read_jsonl(path)}
        pool = {
            key: set().union(*(v[key] for other, v in vocabulary.items() if other != cid))
            for key in ("entities", "actions")
        }
        frames = keyframe_paths(visual) if graphs[cid] else {}
        rows[cid], pending[cid] = [], 0
        for scene_idx, graph in sorted(graphs[cid].items()):
            claims = graph_claims(graph)
            saved = cached.get(scene_idx)
            if saved and _claim_keys(saved["claims"]) == _claim_keys(claims):
                saved.pop("graph_hash", None)
                rows[cid].append(saved)
                continue
            foils = foil_claims(
                graph, pool, seed=f"{settings['foil_seed']}:{cid}:{scene_idx}",
                entities=settings["foil_entities"], actions=settings["foil_actions"],
            )
            questions = {
                claim["id"]: q for claim in [*claims, *foils] if (q := question(claim)) is not None
            }
            pending[cid] += 1
            tasks.append({
                "task_id": f"{cid}:{scene_idx}", "state": STATE, "images": frames[scene_idx],
                "questions": questions,
                "_meta": (cid, scene_idx, claims, foils),
            })
    meta = {task["task_id"]: task.pop("_meta") for task in tasks}
    reused = sum(len(v) for v in rows.values())

    def publish(cid):
        target = output_dir / f"{cid}.jsonl"
        if rows[cid]:
            atomic_write_jsonl(
                target, sorted(rows[cid], key=lambda row: row["scene_idx"]),
                durable=True, sort_keys=False,
            )
        else:
            target.unlink(missing_ok=True)

    failures = 0
    with tqdm(total=len(tasks), desc=f"score {scene_arm(model)}", unit="scene") as bar:

        def receive(event):
            nonlocal failures
            cid, scene_idx, claims, foils = meta[event["task_id"]]
            if "error" in event:
                failures += 1
                tqdm.write(f"[d3] {event['task_id']} failed: {event['error']}")
            else:
                for claim in [*claims, *foils]:
                    claim.update(claim_score(claim, event["answers"].get(claim["id"])))
                rows[cid].append({
                    "content_id": cid, "scene_idx": scene_idx,
                    "scores": unit_scores(claims),
                    "claims": [_stored_claim(claim) for claim in claims],
                    "foils": [_stored_claim(claim) for claim in foils],
                    "input_tokens": event.get("input_tokens"),
                })
            pending[cid] -= 1
            if pending[cid] == 0:
                publish(cid)
            bar.update(1)

        output_dir.mkdir(parents=True, exist_ok=True)
        if tasks:
            with open_judge(context, settings, server=d3_server, log=tqdm.write) as pool:
                pool.run(tasks, receive)
    for cid in rows:
        if pending[cid] == 0:
            publish(cid)
    atomic_write_json(
        output_dir / "scoring_identity.json",
        {**identity, "backend": d3_server or "in-process", "videos": len(visuals), "stored_scene_rows": stored_rows,
         "usable_scenes": sum(len(v) for v in graphs.values()),
         "scored_scenes": sum(len(v) for v in rows.values()), "reused_scenes": reused},
        durable=True,
    )
    print(f"[SCORE] videos={len(visuals)} scenes={stored_rows} "
          f"usable={sum(len(v) for v in graphs.values())} scored_now={len(tasks) - failures} "
          f"reused={reused} failed={failures}", flush=True)
    return {"failure_count": failures}
