from collections import Counter

import pytest

from preparation.download import format_selector
from preparation.manifest import (
    ManifestContractError, category_of, manifest_rows_from_txt, video_id_from_url,
)
from preparation.split import assign_splits
from visual_sampling import build_fixed_windows

RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def test_manifest_from_txt(tmp_path):
    source = tmp_path / "list.txt"
    source.write_text(
        "[News]\n001_News001 https://www.youtube.com/watch?v=VV3qIkq5ofY\n\n"
        "[Tech]\n002_Tech001 https://youtu.be/NeqB-O1M7IY\n",
        encoding="utf-8",
    )
    rows = manifest_rows_from_txt(source)
    assert [row["content_id"] for row in rows] == [
        "001_News001_VV3qIkq5ofY", "002_Tech001_NeqB-O1M7IY"
    ]
    assert category_of(rows[1]["content_id"]) == "Tech"


def test_manifest_rejects_wrong_section_and_bad_lines(tmp_path):
    source = tmp_path / "list.txt"
    source.write_text("[News]\n001_Tech001 https://youtu.be/abc\n", encoding="utf-8")
    with pytest.raises(ManifestContractError):
        manifest_rows_from_txt(source)
    source.write_text("[News]\n002_News001 https://youtu.be/abc\n", encoding="utf-8")
    with pytest.raises(ManifestContractError):  # the list index must start at 001
        manifest_rows_from_txt(source)
    source.write_text("[News]\nnot a video line\n", encoding="utf-8")
    with pytest.raises(ManifestContractError):
        manifest_rows_from_txt(source)


def test_video_id_from_url_forms():
    assert video_id_from_url("https://www.youtube.com/shorts/abc123") == "abc123"
    assert video_id_from_url("https://www.youtube.com/watch?v=xyz&t=3") == "xyz"
    assert video_id_from_url("https://example.com/") == ""


def test_split_is_stratified_and_keeps_existing_assignments():
    ids = [f"{k * 40 + i + 1:03d}_{category}{i + 1:03d}_v{i}"
           for k, category in enumerate(("News", "Tech")) for i in range(40)]
    first = assign_splits(ids, {}, seed=42, ratios=RATIOS)
    for category in ("News", "Tech"):
        counts = Counter(split for cid, split in first.items() if category_of(cid) == category)
        assert counts == {"train": 32, "val": 4, "test": 4}
    assert assign_splits(ids, {}, seed=42, ratios=RATIOS) == first
    more = ids + [f"{81 + i:03d}_News{41 + i:03d}_n{i}" for i in range(10)]
    second = assign_splits(more, first, seed=42, ratios=RATIOS)
    assert all(second[cid] == first[cid] for cid in ids)
    assert Counter(second[cid] for cid in more if category_of(cid) == "News") == {
        "train": 40, "val": 5, "test": 5
    }


def test_format_selector_prefers_requested_height():
    assert format_selector(360).startswith("bv[ext=mp4][height=360]/")


def test_six_keyframes_per_thirty_second_window():
    windows = build_fixed_windows(65, scene_duration=30, num_keyframes=6)
    assert [w["keyframe_timestamps"] for w in windows[:2]] == [
        [2.5, 7.5, 12.5, 17.5, 22.5, 27.5], [32.5, 37.5, 42.5, 47.5, 52.5, 57.5]
    ]
    assert windows[2]["keyframe_timestamps"] == [62.5]


def test_ytdlp_command_for_a_corporate_proxy(tmp_path):
    from pathlib import Path

    from preparation.download import (
        DOWNLOAD_DEFAULTS, shown_proxy, ytdlp_command, ytdlp_env,
    )

    ca = tmp_path / "ca.crt"
    ca.write_text("cert")
    video, info = Path("/d/videos/v.mp4"), Path("/d/metadata/v")
    plain = ytdlp_command("https://youtu.be/x", video, info, dict(DOWNLOAD_DEFAULTS))
    assert "--proxy" not in plain and "--no-check-certificates" not in plain
    assert plain[-1] == "https://youtu.be/x"
    assert "/d/videos/v.%(ext)s" in plain

    settings = {**DOWNLOAD_DEFAULTS, "proxy": "http://p:8080", "ca_cert_file": str(ca)}
    command = ytdlp_command("https://youtu.be/x", video, info, settings)
    assert command[command.index("--proxy") + 1] == "http://p:8080"
    assert command[command.index("--compat-options") + 1] == "no-certifi"
    env = ytdlp_env(settings)
    assert env["HTTPS_PROXY"] == "http://p:8080" and env["SSL_CERT_FILE"] == str(ca)

    insecure = ytdlp_command("u", video, info, {**settings, "no_check_certificates": True})
    assert "--no-check-certificates" in insecure and "--compat-options" not in insecure
    assert shown_proxy("http://user:pw@p:8080") == "http://***@p:8080"
