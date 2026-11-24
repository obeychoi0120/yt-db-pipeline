from __future__ import annotations

import argparse
import sys

from pipeline_runtime import RunContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score scene graphs with a d3 decision model.")
    parser.add_argument("step", choices=("score", "coverage", "report"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", help="score: the extracted model to score.")
    parser.add_argument("--models", nargs="+", help="coverage/report: the models to compare.")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="score: score every scene again.")
    parser.add_argument("--d3-server", metavar="URL",
                        help="score/coverage: send d3 questions to a vllm-sr server "
                             "(e.g. http://localhost:8899) instead of loading d3 on each GPU.")
    args = parser.parse_args(argv)
    try:
        context = RunContext.load(args.run_id)
        if args.step == "score":
            if not args.model:
                raise ValueError("score requires --model")
            from scoring.steps import score_graphs

            result = score_graphs(context, model=args.model, split=args.split, limit=args.limit,
                                  force=args.force, d3_server=args.d3_server)
        elif args.step == "coverage":
            from scoring.coverage import coverage

            result = coverage(context, models=args.models or [], split=args.split,
                              limit=args.limit, d3_server=args.d3_server)
        else:
            if not args.models:
                raise ValueError("report requires --models")
            from scoring.report import write_report

            result = write_report(context, models=args.models, split=args.split)
        if result.get("failure_count", 0):
            raise RuntimeError(f"{result['failure_count']} scene(s) failed; rerun to retry")
    except KeyboardInterrupt:
        print(f"[INTERRUPTED] {args.step}", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {args.step}: {exc}", file=sys.stderr)
        return 1
    return 0
