from __future__ import annotations

import argparse
import sys

from pipeline_runtime import RunContext
from preparation.steps import STEP_HANDLERS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare YouTube videos: manifest, split, download, keyframes."
    )
    parser.add_argument("step", choices=tuple(STEP_HANDLERS))
    parser.add_argument("--list", dest="video_list", help="TXT video list (create-manifest).")
    parser.add_argument("--limit", type=int, help="Only the first N manifest videos.")
    parser.add_argument("--content-id", dest="content_ids", nargs="+", help="Only these videos.")
    parser.add_argument(
        "--force", action="store_true",
        help="make-split: reassign every video. extract-keyframes: extract again.",
    )
    args = parser.parse_args(argv)
    try:
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        result = STEP_HANDLERS[args.step](
            RunContext.load(),
            video_list=args.video_list, limit=args.limit, content_ids=args.content_ids,
            force=args.force,
        )
        if result.get("failure_count", 0):
            raise RuntimeError(f"{result['failure_count']} video(s) failed; rerun to retry")
    except KeyboardInterrupt:
        print(f"[INTERRUPTED] {args.step}", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[FAILED] {args.step}: {exc}", file=sys.stderr)
        return 1
    return 0
