"""Rename downloaded videos and metadata to the content ids of a new video list.

Files are matched by YouTube video id, so 001_News001_<id>.mp4 becomes
News_and_Public_Affairs_001_<id>.mp4. Nothing is renamed without --apply.

    PYTHONPATH=src python -m preparation.rename_videos --list $LIST --data-root $DATA_ROOT
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from artifact_io import write_json
from preparation.manifest import manifest_rows_from_list, video_id_from_url

OLD_NAME = re.compile(r"\d{3}_[A-Za-z]+\d{3}_(.+)")


def plan_renames(folder: Path, suffix: str, new_ids: dict[str, str]):
    """(source, target) pairs, plus the files that have no entry in the new list."""
    renames, unlisted = [], []
    for path in sorted(folder.glob(f"*{suffix}")):
        stem = path.name[: -len(suffix)]
        if stem in new_ids.values():
            continue  # already renamed
        match = OLD_NAME.fullmatch(stem)
        target_id = new_ids.get(match.group(1)) if match else None
        if target_id is None:
            unlisted.append(path)
        else:
            renames.append((path, folder / f"{target_id}{suffix}"))
    return renames, unlisted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", required=True, help="New TXT or CSV video list.")
    parser.add_argument("--data-root", required=True, help="Folder with videos/ and metadata/.")
    parser.add_argument("--apply", action="store_true", help="Rename; without it only print.")
    args = parser.parse_args()

    new_ids = {video_id_from_url(row["url"]): row["content_id"]
               for row in manifest_rows_from_list(args.list)}
    root = Path(args.data_root)
    for name, suffix in (("videos", ".mp4"), ("metadata", ".json")):
        renames, unlisted = plan_renames(root / name, suffix, new_ids)
        blocked = [(source, target) for source, target in renames if target.exists()]
        renames = [pair for pair in renames if pair not in blocked]
        for source, target in renames:
            print(f"{source.name} -> {target.name}")
            if not args.apply:
                continue
            source.rename(target)
            if suffix == ".json":  # the metadata file also stores its content id
                data = json.loads(target.read_text(encoding="utf-8"))
                data["content_id"] = target.name[: -len(suffix)]
                write_json(target, data)
        for source, target in blocked:
            print(f"[SKIP] {source.name}: {target.name} already exists")
        for path in unlisted:
            print(f"[SKIP] {path.name}: not in the new list")
        done = "renamed" if args.apply else "to rename"
        print(f"{name}: {len(renames)} {done}, {len(blocked)} blocked, "
              f"{len(unlisted)} not in the list\n")
    if not args.apply:
        print("Nothing was renamed. Add --apply to rename.")


if __name__ == "__main__":
    main()
