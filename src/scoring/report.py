"""Aggregate scene scores to videos and models, and write a Markdown report."""

from __future__ import annotations

import json

import numpy as np

from artifact_io import atomic_write_json, read_json, read_jsonl
from pipeline_logging import log_step_start
from scoring.claims import UNITS


def load_rows(directory) -> list[dict]:
    return [row for path in sorted(directory.glob("*.jsonl")) for row in read_jsonl(path)]


def video_means(rows, value) -> dict[str, float]:
    """content_id -> mean of value(row) over the video's scenes that have one."""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        number = value(row)
        if number is not None:
            grouped.setdefault(row["content_id"], []).append(number)
    return {cid: sum(values) / len(values) for cid, values in grouped.items()}


def bootstrap_interval(values, *, samples=2000, seed=0, alpha=0.05):
    """Mean with a percentile interval from resampling videos."""
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return None
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "low": None, "high": None, "videos": int(len(values))}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": mean, "low": float(low), "high": float(high), "videos": int(len(values))}


def split_content_ids(context, split) -> set[str] | None:
    """Videos of one split, or None for every video."""
    if split is None:
        return None
    from preparation.manifest import read_splits

    return {cid for cid, name in read_splits(context.splits_path).items() if name == split}


def model_summary(context, model, split=None) -> dict:
    directory = context.scoring_dir(model)
    rows = load_rows(directory)
    selected = split_content_ids(context, split)
    if selected is not None:
        rows = [row for row in rows if row["content_id"] in selected]
    if not rows:
        where = f" in the {split} split" if split else ""
        raise RuntimeError(f"no scores for {model}{where}; run python -m scoring score first")
    # Count the extracted scenes of the same videos; scoring_identity.json only describes
    # the last scoring call, which may have covered another split.
    from scoring.steps import usable_scenes

    stored = usable = 0
    scene_dir = context.scene_dir(model)
    for path in sorted(scene_dir.glob("*.jsonl")):
        if selected is None or path.stem in selected:
            graphs, count = usable_scenes(path)
            stored, usable = stored + count, usable + len(graphs)
    summary = {
        "scored_scenes": len(rows),
        "stored_scene_rows": stored,
        "usable_scenes": usable,
        "units": {}, "per_video": {},
    }
    for unit in UNITS:
        per_video = video_means(rows, lambda row: (row["scores"].get(unit) or {}).get("score"))
        summary["per_video"][unit] = per_video
        summary["units"][unit] = bootstrap_interval(per_video.values())
    claims = [claim for row in rows for claim in row["claims"]]
    # A role link names an entity or states that there is none; the two behave differently.
    summary["roles"] = {}
    for name, wanted in (("linked", False), ("none", True)):
        values = [claim["effective"] for claim in claims if claim["unit"] == "role"
                  and (claim["value"] == "none") == wanted and claim.get("effective") is not None]
        if values:
            summary["roles"][name] = {"count": len(values), "mean": sum(values) / len(values)}
    abstain = [claim["abstain"] for claim in claims if "abstain" in claim]
    summary["claims"] = len(claims)
    summary["claims_per_scene"] = len(claims) / len(rows)
    summary["abstain_mean"] = sum(abstain) / len(abstain) if abstain else None
    summary["foils"] = {}
    for kind in ("entity", "action"):
        values = [f["p"] for row in rows for f in row["foils"]
                  if f["kind"] == kind and f.get("p") is not None]
        if values:
            summary["foils"][kind] = {
                "count": len(values), "mean_p": sum(values) / len(values),
                "accepted": sum(v >= 0.5 for v in values) / len(values),
            }
    lowest = sorted(
        ((claim["effective"], row["content_id"], row["scene_idx"], claim)
         for row in rows for claim in row["claims"] if claim.get("effective") is not None),
        key=lambda item: item[0],
    )[:15]
    summary["lowest_claims"] = [
        {"content_id": cid, "scene_idx": idx, "kind": claim["kind"], "value": claim["value"],
         "effective": round(score, 3), **({"actor": claim["actor"]} if claim.get("actor") else {}),
         **({"action": claim["action"], "value": claim.get("value_name", claim["value"])}
            if claim["unit"] == "role" else {}),
         **({"entity": claim["entity"]} if claim.get("entity") else {})}
        for score, cid, idx, claim in lowest
    ]
    return summary


def paired_difference(first, second) -> dict:
    """Per-unit mean difference (first - second) over the videos both models scored."""
    result = {}
    for unit in UNITS:
        a, b = first["per_video"][unit], second["per_video"][unit]
        shared = sorted(set(a) & set(b))
        result[unit] = bootstrap_interval([a[cid] - b[cid] for cid in shared])
    return result


def _cell(interval, signed=False):
    if interval is None:
        return "–"
    sign = "+" if signed else ""
    text = f"{interval['mean']:{sign}.3f}"
    if interval["low"] is not None:
        text += f" ({interval['low']:{sign}.3f} ~ {interval['high']:{sign}.3f})"
    return text


def write_report(context, *, models, split=None):
    log_step_start(context, "report", models=models, output_dir=context.reports_dir,
                   **({"split": split} if split else {}))
    summaries = {model: model_summary(context, model, split) for model in models}
    coverage_path = context.run_root / "scoring" / "coverage" / "summary.json"
    coverage = read_json(coverage_path) if coverage_path.is_file() else None
    lines = [
        f"# d3 채점 보고서 ({context.run_id}" + (f", {split}" if split else "") + ")", "",
        "장면 그래프를 주장 단위로 나눠 d3 Decision 모델이 키프레임과 대조한 결과입니다. "
        "점수는 주장이 맞을 확률의 평균이고, 하위 주장은 상위 주장의 점수를 곱해 반영했습니다. "
        "괄호는 영상 단위 bootstrap 95% 구간입니다(영상 2개 미만이면 생략).", "",
        "## 단위별 정밀도", "",
        "| 모델 | Context | Entity | Action | 역할 연결 |", "| --- | --- | --- | --- | --- |",
    ]
    for model, summary in summaries.items():
        lines.append(f"| {model} | " + " | ".join(_cell(summary["units"][u]) for u in UNITS) + " |")
    lines += ["", "## 채점 범위", "",
              "| 모델 | 저장된 장면 | 사용 가능 장면 | 채점한 장면 | 장면당 주장 수 | 평균 기권 확률 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for model, s in summaries.items():
        abstain = "–" if s["abstain_mean"] is None else f"{s['abstain_mean']:.3f}"
        lines.append(f"| {model} | {s['stored_scene_rows']} | {s['usable_scenes']} | "
                     f"{s['scored_scenes']} | {s['claims_per_scene']:.1f} | {abstain} |")
    lines += ["", "사용 가능 장면은 파싱에 성공한 장면입니다. 파싱 실패 장면은 채점하지 않으므로, "
              "위 정밀도는 사용 가능 장면만의 값입니다.", "",
              "## 미끼 주장 (다른 영상에서 가져온 주장)", "",
              "| 모델 | 종류 | 개수 | 평균 지지 확률 | 0.5 이상 비율 |", "| --- | --- | --- | --- | --- |"]
    for model, s in summaries.items():
        for kind, foil in s["foils"].items():
            lines.append(f"| {model} | {kind} | {foil['count']} | {foil['mean_p']:.3f} | "
                         f"{foil['accepted']:.3f} |")
    lines += ["", "미끼는 맞지 않아야 하는 주장이므로 값이 낮을수록 Judge가 믿을 만합니다. "
              "우연히 실제로 화면에 있는 대상이 뽑힐 수 있어 0이 되지는 않습니다.", ""]
    lines += ["## 역할 연결 세부", "",
              "| 모델 | 엔티티를 지정한 연결 | 평균 | `none`이라고 한 연결 | 평균 |",
              "| --- | --- | --- | --- | --- |"]
    for model, s in summaries.items():
        cells = []
        for name in ("linked", "none"):
            item = s["roles"].get(name)
            cells += ["0", "–"] if item is None else [str(item["count"]), f"{item['mean']:.3f}"]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")
    differences = {}
    if len(models) >= 2:
        lines += ["## 같은 영상 쌍 비교", "",
                  "| 비교 | Context | Entity | Action | 역할 연결 |", "| --- | --- | --- | --- | --- |"]
        for index, first in enumerate(models):
            for second in models[index + 1:]:
                diff = paired_difference(summaries[first], summaries[second])
                differences[f"{first} - {second}"] = diff
                lines.append(f"| {first} − {second} | "
                             + " | ".join(_cell(diff[u], signed=True) for u in UNITS) + " |")
        lines.append("")
    if coverage and set(models) <= set(coverage["models"]):
        lines += ["## 재현율 (합집합 기준)", "",
                  f"비교한 모델들의 주장 중 유효 점수 {coverage['threshold']} 이상인 엔티티·행동·토픽을 "
                  "모아 기준으로 삼고, 각 모델의 그래프가 이를 담았는지 d3가 판정했습니다.", "",
                  "| 모델 | 장면 수 | 평균 재현율 |", "| --- | --- | --- |"]
        for model in models:
            item = coverage["recall"][model]
            mean = "–" if item["mean"] is None else f"{item['mean']:.3f}"
            lines.append(f"| {model} | {item['scenes']} | {mean} |")
        lines.append("")
    for model, s in summaries.items():
        lines += [f"## 점수가 낮은 주장: {model}", "",
                  "| 영상 | 장면 | 종류 | 주장 | 유효 점수 |", "| --- | --- | --- | --- | --- |"]
        for claim in s["lowest_claims"]:
            subject = claim.get("entity") or claim.get("actor")
            text = f"{subject}: {claim['value']}" if subject else str(claim["value"])
            if "action" in claim:
                text = f"{subject or 'unknown'} / {claim['action']} → {claim['value']}"
            lines.append(f"| {claim['content_id']} | {claim['scene_idx']} | {claim['kind']} | "
                         f"{text} | {claim['effective']:.3f} |")
        lines.append("")
    name = "scoring_" + "__".join(models) + (f"_{split}" if split else "")
    context.reports_dir.mkdir(parents=True, exist_ok=True)
    (context.reports_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    for summary in summaries.values():
        summary.pop("per_video")
    atomic_write_json(context.reports_dir / f"{name}.json",
                      {"models": summaries, "paired": differences, "coverage": coverage},
                      durable=True)
    print(f"[REPORT] {context.reports_dir / (name + '.md')}", flush=True)
    print(json.dumps({m: {u: s["units"][u] and round(s["units"][u]["mean"], 3) for u in UNITS}
                      for m, s in summaries.items()}), flush=True)
    return {"failure_count": 0}
