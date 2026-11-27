"""Video list (TXT or CSV), two-column manifest and video-level splits."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

MANIFEST_FIELDS = ("content_id", "url")
SPLIT_FIELDS = ("content_id", "category", "split")
SPLITS = ("train", "val", "test")


class ManifestContractError(ValueError):
    """A list, manifest or split file violates its contract."""


CATEGORY = r"[A-Za-z]+(?:_[A-Za-z]+)*"
NUMBERED_LINE = re.compile(r"(\d{3})_([A-Za-z]+)(\d{3})\s+(https?://\S+)")
NAMED_LINE = re.compile(rf"({CATEGORY})_(\d{{3}})(?:\.mp4)?\s+(https?://\S+)")


# Columns a CSV video list must have; the others (title, query, ...) are ignored.
CSV_LIST_FIELDS = ("Name", "URL")


def _list_lines(path: Path) -> list[tuple[int, str]]:
    """``(line number, "<name> <url>")`` of a TXT list, or of the Name and URL columns of a CSV."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        if path.suffix.lower() != ".csv":
            return list(enumerate(handle, start=1))
        reader = csv.DictReader(handle)
        missing = [field for field in CSV_LIST_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ManifestContractError(f"{path} has no {', '.join(missing)} column")
        return [
            (line_number, f"{(row['Name'] or '').strip()} {(row['URL'] or '').strip()}")
            for line_number, row in enumerate(reader, start=2)
            if (row["Name"] or "").strip() or (row["URL"] or "").strip()
        ]


def read_video_list(list_path: str | Path) -> list[tuple[str, str]]:
    """Read a video list: a TXT of ``<name> <url>`` lines (``[Category]`` headers optional) or a
    CSV with Name and URL columns.

    A name is ``Category_NNN.mp4`` (NNN counts within the category, and the list id drops
    ``.mp4``) or the older ``NNN_CategoryNNN`` (the first number counts the whole list).
    """
    path = Path(list_path)
    current_category = ""
    rows: list[tuple[str, str]] = []
    seen_list_ids: set[str] = set()
    per_category: dict[str, int] = {}
    for line_number, raw_line in _list_lines(path):
        line = raw_line.strip()
        if not line:
            continue
        category_match = re.fullmatch(rf"\[({CATEGORY})\]", line)
        if category_match:
            current_category = category_match.group(1)
            continue
        if numbered := NUMBERED_LINE.fullmatch(line):
            index, item_category, item_number, url = numbered.groups()
            list_id = f"{index}_{item_category}{item_number}"
            if int(index) != len(rows) + 1:
                raise ManifestContractError(
                    f"line {line_number} index must be {len(rows) + 1:03d}: {list_id}"
                )
        elif named := NAMED_LINE.fullmatch(line):
            item_category, item_number, url = named.groups()
            list_id = f"{item_category}_{item_number}"
            expected = per_category.get(item_category, 0) + 1
            if int(item_number) != expected:
                raise ManifestContractError(
                    f"line {line_number} number must be {expected:03d}: {list_id}"
                )
            per_category[item_category] = expected
        else:
            raise ManifestContractError(f"invalid video list line {line_number}: {line}")
        if current_category and item_category != current_category:
            raise ManifestContractError(
                f"line {line_number} list id does not match [{current_category}]: {list_id}"
            )
        if list_id in seen_list_ids:
            raise ManifestContractError(f"duplicate list id on line {line_number}: {list_id}")
        seen_list_ids.add(list_id)
        rows.append((list_id, url))
    return rows


def video_id_from_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.strip("/").split("/")[0]
    query_video_id = parse_qs(parsed.query).get("v", [""])[0]
    if query_video_id:
        return query_video_id
    path_parts = [part for part in parsed.path.split("/") if part]
    for marker in ("shorts", "embed", "live"):
        if marker in path_parts:
            index = path_parts.index(marker)
            if index + 1 < len(path_parts):
                return path_parts[index + 1]
    return ""


def normalize_content_id(value: object) -> str:
    text = str(value or "").strip()
    keep = []
    for char in text:
        if char.isascii() and (char.isalnum() or char in ("-", "_", ".")):
            keep.append(char)
        elif char.isspace():
            keep.append("_")
    normalized = "".join(keep).strip("._")
    if not normalized:
        raise ManifestContractError(f"content_id has no filesystem-safe characters: {text!r}")
    return normalized


def category_of(content_id: str) -> str:
    """Sports_001_<video id> -> Sports, and the older 001_News001_<video id> -> News."""
    match = re.match(rf"\d{{3}}_([A-Za-z]+)\d{{3}}_|({CATEGORY})_\d{{3}}_", content_id)
    if not match:
        raise ManifestContractError(
            f"content_id does not start with Category_NNN_ or NNN_CategoryNNN_: {content_id}"
        )
    return match.group(1) or match.group(2)


def manifest_rows_from_list(list_path: str | Path) -> list[dict[str, str]]:
    rows = []
    for list_id, url in read_video_list(list_path):
        video_id = video_id_from_url(url)
        if not video_id:
            raise ManifestContractError(f"no YouTube video id in URL for {list_id}: {url}")
        rows.append({"content_id": normalize_content_id(f"{list_id}_{video_id}"), "url": url})
    return validate_manifest_rows(rows, source=str(list_path))


def validate_manifest_rows(
    rows: Iterable[dict[str, Any]], *, source: str = "manifest"
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen_content_ids: set[str] = set()
    seen_urls: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        if not isinstance(row, dict) or set(row) != set(MANIFEST_FIELDS):
            raise ManifestContractError(
                f"{source} line {line_number} must contain exactly {','.join(MANIFEST_FIELDS)}"
            )
        content_id = str(row.get("content_id") or "").strip()
        url = str(row.get("url") or "").strip()
        if not content_id or not url:
            raise ManifestContractError(f"{source} line {line_number} has an empty field")
        if content_id in seen_content_ids:
            raise ManifestContractError(
                f"{source} line {line_number} duplicates content_id {content_id}"
            )
        if url in seen_urls:
            raise ManifestContractError(f"{source} line {line_number} duplicates url {url}")
        seen_content_ids.add(content_id)
        seen_urls.add(url)
        normalized.append({"content_id": content_id, "url": url})
    if not normalized:
        raise ManifestContractError(f"{source} contains no rows")
    return normalized


def _read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ManifestContractError(f"failed to read {path}: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != list(fields):
        raise ManifestContractError(f"{path} columns must be exactly {','.join(fields)}")
    return list(reader)


def write_csv(path: str | Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)


def read_manifest_rows(path: str | Path) -> list[dict[str, str]]:
    return validate_manifest_rows(_read_csv(Path(path), MANIFEST_FIELDS), source=str(path))


def read_splits(path: str | Path) -> dict[str, str]:
    """content_id -> split of every assigned video."""
    splits: dict[str, str] = {}
    for row in _read_csv(Path(path), SPLIT_FIELDS):
        if row["split"] not in SPLITS or row["content_id"] in splits:
            raise ManifestContractError(f"invalid or duplicate split row: {row}")
        splits[row["content_id"]] = row["split"]
    return splits


def selected_content_ids(context, *, split: str | None = None, limit: int | None = None):
    """Manifest order, optionally one split and the first ``limit`` videos."""
    if not context.manifest_path.is_file():
        raise ManifestContractError(
            f"manifest not found: {context.manifest_path}; run python -m preparation make-split"
        )
    content_ids = [row["content_id"] for row in read_manifest_rows(context.manifest_path)]
    if split is not None:
        if split not in SPLITS:
            raise ManifestContractError(f"split must be one of {', '.join(SPLITS)}")
        if not context.splits_path.is_file():
            raise ManifestContractError(
                f"splits not found: {context.splits_path}; run python -m preparation make-split"
            )
        splits = read_splits(context.splits_path)
        content_ids = [cid for cid in content_ids if splits.get(cid) == split]
    return content_ids if limit is None else content_ids[:limit]
