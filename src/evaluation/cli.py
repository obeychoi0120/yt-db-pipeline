from __future__ import annotations

import argparse
import sys

from pipeline_runtime import RunContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare student models with the teacher.")
    parser.add_argument("step", choices=("compare",))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--teacher", default="gemini")
    parser.add_argument("--students", nargs="+", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    try:
        from evaluation.steps import compare

        compare(RunContext.load(args.run_id), teacher=args.teacher, students=args.students,
                split=args.split, limit=args.limit)
    except KeyboardInterrupt:
        print(f"[INTERRUPTED] {args.step}", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {args.step}: {exc}", file=sys.stderr)
        return 1
    return 0
