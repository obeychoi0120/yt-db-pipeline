from __future__ import annotations

import argparse
import sys

from pipeline_runtime import RunContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning data and checkpoints.")
    parser.add_argument("step", choices=("build-dataset", "model-view", "merge"))
    parser.add_argument("--run-id")
    parser.add_argument("--teacher", default="gemini", help="build-dataset: the teacher model.")
    parser.add_argument("--schema", help="build-dataset: the prompt used for extraction.")
    parser.add_argument("--min-scene-score", type=float,
                        help="build-dataset: drop teacher scenes scored below this.")
    parser.add_argument("--allow-prompt-mismatch", action="store_true")
    parser.add_argument("--model", default="gemma4_e2b", help="model-view: the student model.")
    parser.add_argument("--training-dir", help="merge: the ms-swift output directory.")
    parser.add_argument("--output", help="model-view / merge: the directory to create.")
    args = parser.parse_args(argv)
    try:
        context = RunContext.load(args.run_id)
        if args.step == "build-dataset":
            if not args.schema:
                raise ValueError("build-dataset requires --schema")
            from finetuning.dataset import build_dataset

            build_dataset(context, teacher=args.teacher, schema=args.schema,
                          min_scene_score=args.min_scene_score,
                          allow_prompt_mismatch=args.allow_prompt_mismatch)
        elif args.step == "model-view":
            if not args.output:
                raise ValueError("model-view requires --output")
            from finetuning.model_view import build_model_view

            build_model_view(context, model=args.model, output=args.output)
        else:
            if not (args.training_dir and args.output):
                raise ValueError("merge requires --training-dir and --output")
            from finetuning.merge import merge_best

            merge_best(context, training_dir=args.training_dir, output=args.output)
    except KeyboardInterrupt:
        print(f"[INTERRUPTED] {args.step}", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {args.step}: {exc}", file=sys.stderr)
        return 1
    return 0
