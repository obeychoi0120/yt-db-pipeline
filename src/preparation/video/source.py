from pathlib import Path
from typing import Any

SOURCE_IDENTITY_KEYS = (
    "content_id",
    "source_video_path",
    "source_file_size",
    "source_mtime_ns",
)


def source_matches_inventory(row: dict[str, Any]) -> bool:
    try:
        source = Path(row["source_video_path"])
        stat = source.stat()
        return (
            source.is_file()
            and stat.st_size == row["source_file_size"]
            and stat.st_mtime_ns == row["source_mtime_ns"]
        )
    except (OSError, TypeError, KeyError):
        return False
