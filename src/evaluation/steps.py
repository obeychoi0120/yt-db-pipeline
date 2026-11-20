from __future__ import annotations

from artifact_io import atomic_write_json, read_json
from evaluation.agreement import compare_graphs, empty_agreement
from extraction.step_support import visual_rows
from pipeline_logging import log_step_start
from scoring.claims import UNITS
from scoring.report import bootstrap_interval, model_summary, video_means
from scoring.steps import usable_scenes

DEFAULTS = {"match_threshold": 0.8}
FIELDS = (
    ("medium", "medium 일치", lambda r: r["medium"]),
    ("format", "format 일치", lambda r: r["format"]),
    ("topics_f1", "topics F1", lambda r: r["topics"]["f1"]),
    ("entities_f1", "Entity F1", lambda r: r["entities"]["f1"]),
    ("actions_f1", "Action F1", lambda r: r["actions"]["f1"]),
    ("roles", "역할 연결 일치", lambda r: r["roles"]["agreement"]),
)


def evaluation_settings(context) -> dict:
    return {**DEFAULTS, **(context.config.get("evaluation") or {})}


def _cell(interval):
    if interval is None:
        return "–"
    text = f"{interval['mean']:.3f}"
    if interval["low"] is not None:
        text += f" ({interval['low']:.3f} ~ {interval['high']:.3f})"
    return text


def health(context, model) -> dict | None:
    path = context.scene_dir(model) / "generation_stats.json"
    return read_json(path) if path.is_file() else None


def compare(context, *, teacher, students, split=None, limit=None, encoder=None):
    settings = evaluation_settings(context)
    threshold = settings["match_threshold"]
    log_step_start(context, "compare", teacher=teacher, students=students,
                   match_threshold=threshold, output_dir=context.reports_dir,
                   **({"split": split} if split else {}), **({"limit": limit} if limit else {}))
    if encoder is None:
        from evaluation.encoder import TextEncoder

        encoder = TextEncoder(context.model_path("bge"))
    visuals = visual_rows(context, split=split, limit=limit)
    models = [teacher, *students]
    agreement = {}
    for student in models:  # the teacher against itself checks the measure (must be 1)
        rows = []
        for visual in visuals:
            cid = visual["content_id"]
            reference, _ = usable_scenes(context.scene_dir(teacher) / f"{cid}.jsonl")
            graphs, _ = usable_scenes(context.scene_dir(student) / f"{cid}.jsonl")
            for scene_idx, graph in sorted(reference.items()):
                row = (compare_graphs(graphs[scene_idx], graph, encoder.encode, threshold=threshold)
                       if scene_idx in graphs else empty_agreement(graph))
                rows.append({"content_id": cid, "scene_idx": scene_idx, **row})
        if not rows:
            raise RuntimeError(f"no teacher scenes to compare for {teacher}")
        agreement[student] = {
            "scenes": len(rows), "missing": sum(bool(row.get("missing")) for row in rows),
            "fields": {key: bootstrap_interval(video_means(rows, value).values())
                       for key, _, value in FIELDS},
        }
    lines = [
        f"# LoRA 평가 보고서 ({context.run_id})", "",
        f"교사 `{teacher}`의 사용 가능 장면을 기준으로 비교했습니다"
        + (f" (split: {split})." if split else "."),
        "괄호는 영상 단위 bootstrap 95% 구간입니다(영상 2개 미만이면 생략).", "",
        "## 교사 일치도", "",
        f"topics·엔티티·행동은 BGE 임베딩 유사도 {threshold} 이상인 쌍을 일대일로 짝지어 계산했습니다. "
        "학생에게 사용 가능한 그래프가 없는 장면은 일치 0으로 넣었습니다.", "",
        "| 모델 | 장면 | 그래프 없음 | " + " | ".join(label for _, label, _ in FIELDS) + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in FIELDS) + " |",
    ]
    for model in models:
        item = agreement[model]
        lines.append(f"| {model} | {item['scenes']} | {item['missing']} | "
                     + " | ".join(_cell(item["fields"][key]) for key, _, _ in FIELDS) + " |")
    lines += ["", f"`{teacher}` 행은 교사를 자기 자신과 비교한 값으로, 측정이 맞게 동작하면 모두 1입니다.", "",
              "## 형식 건전성", "",
              "| 모델 | 전체 | 성공 | 반복 중단 | 파싱 실패 | 토큰 한도 | 오류 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    stats = {model: health(context, model) for model in models}
    for model in models:
        s = stats[model]
        lines.append(f"| {model} | –  | – | – | – | – | – |" if s is None else
                     f"| {model} | {s['total']} | {s['success']} | {s['repeat_occurred']} | "
                     f"{s['parse_failed']} | {s['token_limit']} | {s['error']} |")
    lines += ["", "형식 건전성은 마지막 추출 실행 전체의 집계이며 split으로 나누지 않은 값입니다.", "",
              "## d3 점수 (정밀도)", ""]
    scores = {}
    for model in models:
        try:
            scores[model] = model_summary(context, model)
        except (RuntimeError, OSError):
            scores[model] = None
    if any(scores.values()):
        lines += ["| 모델 | 채점한 장면 | Context | Entity | Action | 역할 연결 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for model in models:
            s = scores[model]
            lines.append(f"| {model} | – | – | – | – | – |" if s is None else
                         f"| {model} | {s['scored_scenes']} | "
                         + " | ".join(_cell(s["units"][unit]) for unit in UNITS) + " |")
        lines += ["", "d3 점수는 `python -m scoring score`로 채점해 둔 전체 장면의 값입니다. "
                  "split을 맞추려면 같은 `--split`으로 채점한 Run에서 보고서를 만드세요."]
    else:
        lines.append("채점 결과가 없습니다. `python -m scoring score --model <모델>`을 먼저 실행하세요.")
    lines.append("")
    name = "evaluation_" + "__".join(models) + (f"_{split}" if split else "")
    context.reports_dir.mkdir(parents=True, exist_ok=True)
    (context.reports_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    for summary in scores.values():
        if summary:
            summary.pop("per_video", None)
    atomic_write_json(
        context.reports_dir / f"{name}.json",
        {"teacher": teacher, "split": split, "match_threshold": threshold,
         "agreement": agreement, "health": stats, "d3": scores},
        durable=True,
    )
    print(f"[EVALUATION] {context.reports_dir / (name + '.md')}", flush=True)
    return {"failure_count": 0, "agreement": agreement}
