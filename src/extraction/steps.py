from __future__ import annotations
from functools import partial
from tqdm import tqdm
from artifact_io import atomic_write_json, file_fingerprint
from extraction.backends import GeminiWorkerPool
from extraction.generation import vllm_generator
from extraction.generation_config import generation_settings, penalty_schedule, vllm_settings
from extraction.graph_warnings import OUTCOMES, needs_generation, outcome
from extraction.iab import extract_iab_gt
from extraction.progress import InferenceProgress
from extraction.scene_executor import run_gemini_scenes, run_vllm_scenes
from extraction.scene_storage import (
    ExtractionStepError, ensure_generation_identity, read_scene_records,
)
from extraction.semantic_graph import GRAPH_SCHEMA_VERSION
from extraction.semantic_graph.parser import GRAPH_PARSER_VERSION
from extraction.step_support import result, scene_generation_rows, visual_rows
from pipeline_logging import log_step_start
from pipeline_runtime import scene_arm

# Model entry keys that are not part of the recorded model identity: how requests are sent, and
# the output limit that the generation settings record on their own.
RUNTIME_KEYS = ("backend", "threads", "scene_max_new_tokens")
GEMINI_THREADS = 32


def engine_settings(context, model):
    """vLLM engine arguments: extraction.vllm plus the model's multimodal processor kwargs."""
    settings = dict(context.config["extraction"].get("vllm") or {})
    processor = context.model_settings(model).get("mm_processor_kwargs")
    if processor:
        settings["mm_processor_kwargs"] = dict(processor)
    return vllm_settings(settings)


def prompt_provenance(
    context, schema, model, *, repetition_penalty=None, presence_penalty=None,
    stop_repeated_lines=False,
):
    """Prompt, model and generation settings recorded beside the scenes of one model."""
    path = context.prompt_path(schema)
    local = context.generation_backend(model) == "vllm"
    if not local and (repetition_penalty is not None or presence_penalty is not None
                      or stop_repeated_lines):
        raise ValueError("penalties and repeated-line stopping apply only to local vLLM models")
    extraction = context.config["extraction"]
    settings = {"max_new_tokens": context.max_new_tokens(model)}
    if local:
        generation = generation_settings(
            model, repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
            stop_repeated_lines=stop_repeated_lines,
        )
        settings.update(
            backend="vllm",
            vllm=engine_settings(context, model),
            generation=generation,
            repetition_penalty=(
                penalty_schedule(repetition_penalty) if repetition_penalty is not None
                else [generation["repetition_penalty"]]
            ),
        )
        model_identity = {"path": str(context.model_path(model))}
    else:
        settings["backend"] = "gemini"
        model_identity = {
            key: value for key, value in context.model_settings(model).items()
            if key not in RUNTIME_KEYS
        }
    settings["visual_evidence"] = dict(extraction["visual_evidence"])
    settings["response_parser"] = GRAPH_PARSER_VERSION
    return {
        "scene_arm": scene_arm(model),
        "prompt_path": str(path),
        "prompt_hash": file_fingerprint(path),
        "model": model_identity,
        "settings": settings,
        "schema_contract": GRAPH_SCHEMA_VERSION,
    }


def prepare_scene_batch(visuals, scene_dir, prepare_rows, *, force):
    existing, planned = {}, []
    for visual in visuals:
        cid = visual["content_id"]
        output = scene_dir / f"{cid}.jsonl"
        rows = prepare_rows(visual)
        try:
            cached = (
                read_scene_records(output, include_failures=True)
                if output.is_file() and (not force)
                else []
            )
        except (ValueError, ExtractionStepError) as exc:
            raise ExtractionStepError(
                f"cannot safely identify existing scenes: {output}: {exc}"
            ) from exc
        by_index = {r["scene_idx"]: r for r in cached}
        if len(by_index) != len(cached):
            raise ExtractionStepError(f"duplicate cached scenes: {output}")
        missing = []
        for row in rows:
            saved = by_index.get(row["scene_idx"])
            if needs_generation(saved):
                missing.append(row)
        expected_indices = {row["scene_idx"] for row in rows}
        existing[cid] = [row for row in cached if row["scene_idx"] in expected_indices]
        if missing:
            planned.append(
                (
                    visual,
                    [
                        {
                            "scene_idx": row["scene_idx"],
                            "keyframes": row["keyframes"],
                            "scene_start": row.get("scene_start_seconds"),
                            "scene_end": row.get("scene_end_seconds"),
                        }
                        for row in missing
                    ],
                )
            )
    return existing, planned


def extract_graph_scenes(
    context, *, model, schema, force=False, split=None, limit=None,
    repetition_penalty=None, presence_penalty=None, stop_repeated_lines=False,
):
    stage = "extract-graph-scenes"
    backend = context.generation_backend(model)
    path = context.prompt_path(schema)
    scene_dir = context.scene_dir(model)
    log_step_start(
        context, stage, model=model, model_settings=context.model_settings(model),
        prompt=path, output_dir=scene_dir, max_new_tokens=context.max_new_tokens(model),
        force=force,
        **({"split": split} if split else {}),
        **({"limit": limit} if limit else {}),
        **({"repetition_penalty": repetition_penalty} if repetition_penalty is not None else {}),
        **({"presence_penalty": presence_penalty} if presence_penalty is not None else {}),
        **({"stop_repeated_lines": True} if stop_repeated_lines else {}),
    )
    prompt = path.read_text(encoding="utf-8")
    print("[PREPARE] Reading prompt/model settings and manifest...", flush=True)
    settings = context.config["extraction"]
    provenance = prompt_provenance(
        context, path, model, repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty, stop_repeated_lines=stop_repeated_lines,
    )
    penalties = provenance["settings"].get("repetition_penalty", [1.0])
    ensure_generation_identity(scene_dir, provenance, force=force)
    (scene_dir / "generation_stats.json").unlink(missing_ok=True)
    visuals = visual_rows(context, split=split, limit=limit)
    prepare_rows = partial(
        scene_generation_rows,
        prompt=prompt,
        max_new_tokens=context.max_new_tokens(model),
        generation=provenance["settings"].get("generation", {}),
    )
    print("[PREPARE] Counting pending scenes and checking cached outputs...", flush=True)
    existing, planned = prepare_scene_batch(visuals, scene_dir, prepare_rows, force=force)
    total = sum((len(scenes) for _, scenes in planned))
    reused = sum(
        not needs_generation(row)
        for records in existing.values() for row in records
    )

    def pending_contents():
        for visual, scenes in planned:
            missing = prepare_rows(visual, scenes=scenes)
            yield (visual, missing)

    with InferenceProgress(
        total=total,
        reused=reused,
        desc=scene_arm(model),
        unit="scene",
        outcomes=OUTCOMES,
        progress_factory=tqdm,
    ) as progress:
        if backend == "vllm":
            run_vllm_scenes(
                pending_contents(),
                provenance=provenance,
                scene_dir=scene_dir,
                model_path=context.model_path(model),
                generator_factory=vllm_generator,
                progress=progress,
                existing_records=existing,
                vllm_options=provenance["settings"]["vllm"],
                image_limit=settings["visual_evidence"]["num_keyframes"],
                penalties=penalties,
            )
        else:
            model_settings = context.model_settings(model)
            gemini = {
                key: value for key, value in model_settings.items() if key not in RUNTIME_KEYS
            }
            pool = GeminiWorkerPool(model_settings.get("threads", GEMINI_THREADS), **gemini)
            run_gemini_scenes(
                pending_contents(),
                provenance=provenance,
                pool=pool,
                records_by_content=existing,
                scene_dir=scene_dir,
                progress=progress,
            )
    failure_count = sum("error" in row for records in existing.values() for row in records)
    # Each scene is counted once; the outcomes sum to the total.
    stats = {"total": total + reused, **dict.fromkeys(OUTCOMES, 0)}
    for records in existing.values():
        for record in records:
            stats[outcome(record)] += 1
    atomic_write_json(scene_dir / "generation_stats.json", stats, durable=True)
    return result(f"{stage}-{model}", content_count=len(visuals), failure_count=failure_count)


STEP_HANDLERS = {
    "extract-graph-scenes": extract_graph_scenes,
    "extract-iab-gt": extract_iab_gt,
}
