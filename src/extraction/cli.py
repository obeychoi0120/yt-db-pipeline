from __future__ import annotations

import argparse
import sys

from extraction.generation_config import penalty_schedule, validate_presence_penalty
from extraction.steps import STEP_HANDLERS
from pipeline_runtime import RunContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract scene graphs from prepared keyframes.")
    parser.add_argument("step", choices=tuple(STEP_HANDLERS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--schema", help="extract-graph-scenes: path to one Markdown prompt."
    )
    parser.add_argument(
        "--model",
        help="extract-graph-scenes: a key of config models with backend vllm (local) or gemini.",
    )
    parser.add_argument(
        "--taxonomy", help="extract-iab-gt: path to the tab-separated IAB Content Taxonomy table."
    )
    parser.add_argument("--split", choices=("train", "val", "test"), help="Only this split.")
    parser.add_argument("--limit", type=int, help="Only the first N selected videos.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all scenes for this model, including completed scenes.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        nargs="+",
        help="Local vLLM generated-token repetition penalty in [1, 2] (default 1.05). Several "
        "values run one pass each: the first pass covers every pending scene and each later "
        "pass retries the scenes that did not succeed.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        help="Local vLLM presence penalty in [-2, 2] (default 0).",
    )
    parser.add_argument(
        "--stop-repeated-lines",
        action="store_true",
        help="Local vLLM only: end the output instead of writing a line identical to an "
        "earlier line (the repeated line is dropped). Off by default.",
    )
    args = parser.parse_args(argv)
    try:
        options = {}
        if args.repetition_penalty is not None:
            schedule = penalty_schedule(args.repetition_penalty)
            options["repetition_penalty"] = schedule[0] if len(schedule) == 1 else schedule
        if args.presence_penalty is not None:
            options["presence_penalty"] = validate_presence_penalty(args.presence_penalty)
        if args.stop_repeated_lines:
            options["stop_repeated_lines"] = True
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        if args.step == "extract-iab-gt":
            _require(args, "--taxonomy")
            _reject(args, options, "--schema", "--model")
            options = {"taxonomy": args.taxonomy}
        else:
            _require(args, "--schema", "--model")
            _reject(args, {}, "--taxonomy")
            options.update(schema=args.schema, model=args.model)
        context = RunContext.load(args.run_id)
        result = STEP_HANDLERS[args.step](
            context,
            force=args.force,
            split=args.split,
            limit=args.limit,
            **options,
        )
        if result.get("failure_count", 0):
            raise RuntimeError(
                f"{result['failure_count']} output(s) failed; completed outputs are preserved. "
                "Rerun this step to retry incomplete outputs."
            )
    except KeyboardInterrupt:
        print(f"[INTERRUPTED] {args.step}", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {args.step}: {exc}", file=sys.stderr)
        return 1
    return 0


def _flag_value(args: argparse.Namespace, flag: str):
    return getattr(args, flag.lstrip("-").replace("-", "_"))


def _require(args: argparse.Namespace, *flags: str) -> None:
    missing = [flag for flag in flags if not _flag_value(args, flag)]
    if missing:
        raise ValueError(f"{args.step} requires {', '.join(missing)}")


def _reject(args: argparse.Namespace, generation_options: dict, *flags: str) -> None:
    given = [flag for flag in flags if _flag_value(args, flag)]
    if generation_options:
        given.append("generation penalty options")
    if given:
        raise ValueError(f"{args.step} does not take {', '.join(given)}")
