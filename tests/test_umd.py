"""UMD category labelling: class table, frame sampling, labels, resume and CLI flags."""

import json
from pathlib import Path

import pytest
from PIL import Image

from artifact_io import read_jsonl
from preparation import cli, umd
from visual_sampling import timestamp_stem

ROOT = Path(__file__).resolve().parents[1]
TABLE = (
    "Category,Subcategory Count,Subcategories\n"
    'Sports,2,"Baseball, Soccer",\n'
    'Action_and_Thriller,2,"Law,Crime"\n'
)


class FakeClient:
    """Puts top probability on the first preferred option it is offered."""

    def __init__(self, prefer, top=0.6, fail=False):
        self.prefer, self.top, self.fail, self.calls = prefer, top, fail, []

    def choose(self, images, instructions, options):
        assert all(image["mime_type"] == "image/png" for image in images)
        self.calls.append({"images": len(images), "options": dict(options)})
        if self.fail:
            raise RuntimeError("boom")
        choice = next(key for key in self.prefer if key in options)
        rest = (1 - self.top) / (len(options) - 1)
        probabilities = {key: (self.top if key == choice else rest) for key in options}
        return choice, self.top, probabilities


class Context:
    def __init__(self, root, umd_settings=None, iab_settings=None):
        self.root, self.run_id = root, None
        self.config = {
            "extraction": {"iab": {"api_key": "k", **(iab_settings or {})}},
            "preparation": {"umd": {"threads": 2, **(umd_settings or {})}},
        }
        self.umd_dir = root / "umd_categories"


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "classes.csv"
    path.write_text(TABLE, encoding="utf-8")
    return path


def videos(tmp_path, monkeypatch, frames=(3, 120)):
    """One prepared video per entry, with that many 5-second keyframes in 30-second windows."""
    rows = []
    for index, count in enumerate(frames):
        cid = f"Sports_{index + 1:03d}_abc{index}"
        directory = tmp_path / "frames" / cid
        directory.mkdir(parents=True)
        stamps = [5 * second for second in range(count)]
        for stamp in stamps:
            Image.new("RGB", (8, 8), (stamp % 256, 0, 0)).save(
                directory / f"{timestamp_stem(stamp)}.png"
            )
        windows = [{"keyframe_timestamps": stamps[start:start + 6]}
                   for start in range(0, count, 6)]
        timestamp_json = tmp_path / f"{cid}.json"
        timestamp_json.write_text(json.dumps(windows), encoding="utf-8")
        rows.append({"content_id": cid, "frames_dir": str(directory),
                     "timestamp_json": str(timestamp_json)})
    monkeypatch.setattr(
        umd, "prepared_videos",
        lambda context, content_ids=None, limit=None: [
            row for row in rows if not content_ids or row["content_id"] in content_ids
        ][:limit],
    )
    monkeypatch.setattr(umd, "log_step_start", lambda *args, **kwargs: None)
    return [row["content_id"] for row in rows]


def run(context, table, client, **options):
    return umd.classify_umd_categories(
        context, classes=str(table), client_factory=lambda **_: client, **options
    )


def labels(context):
    return {row["content_id"]: row for row in read_jsonl(context.umd_dir / "labels.jsonl")}


def test_repository_class_table_fits_one_question():
    classes = umd.load_classes(ROOT / "data" / "UMD_class_261127.csv")
    assert len(classes) == 232 and len(set(classes.values())) == 16
    # A missing space after a comma still splits, and a trailing comma adds no option.
    assert [name for name, category in classes.items() if category == "Business_and_Finance"] == [
        "Financial", "Investment", "Stock", "Business"
    ]
    choices = umd.options(classes)
    assert len(choices) == 233 <= umd.MAX_OPTIONS
    assert choices["Law"] == "Law (Action and Thriller)" and umd.NONE in choices


@pytest.mark.parametrize("text, message", [
    ("a,b,c\n", "header"),
    ("Category,Subcategory Count,Subcategories\nSports,3,\"Baseball, Soccer\"\n", "Count"),
    ("Category,Subcategory Count,Subcategories\nA,1,Go\nB,1,Go\n", "duplicate"),
    ("Category,Subcategory Count,Subcategories\nA,2,\"Go, None\"\n", "reserved"),
])
def test_load_classes_rejects_bad_tables(tmp_path, text, message):
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        umd.load_classes(path)


def test_select_frames_spreads_over_the_whole_video():
    frames = [str(index) for index in range(147)]
    chosen = umd.select_frames(frames, 50)
    assert len(chosen) == len(set(chosen)) == 50
    assert chosen[0] == "0" and chosen[-1] == "146"
    assert umd.select_frames(frames[:30], 50) == frames[:30]
    assert umd.select_frames(frames, 1) == ["0"]


def test_step_writes_labels_and_resumes(tmp_path, table, monkeypatch):
    short, long = videos(tmp_path, monkeypatch)
    context = Context(tmp_path, {"max_frames": 50})
    client = FakeClient(prefer=("Soccer",))
    assert run(context, table, client)["failure_count"] == 0
    assert sorted(call["images"] for call in client.calls) == [3, 50]
    assert all(call["options"]["Law"] == "Law (Action and Thriller)" for call in client.calls)

    rows = labels(context)
    assert set(rows[long]) == {
        "content_id", "query_category", "pred_umd_categories", "warning", "frames_total",
        "frames_used",
    }
    assert rows[long]["query_category"] == "Sports"
    assert rows[long]["frames_total"] == 120 and rows[long]["frames_used"] == 50
    assert rows[short]["frames_total"] == rows[short]["frames_used"] == 3
    predictions = rows[long]["pred_umd_categories"]
    assert list(predictions)[0] == "Soccer" and len(predictions) == 5
    assert list(predictions.values()) == sorted(predictions.values(), reverse=True)
    assert rows[long]["warning"] is None
    csv_text = (context.umd_dir / "labels.csv").read_text(encoding="utf-8")
    assert "Soccer:0.6;" in csv_text

    again = FakeClient(prefer=("Law",))
    run(context, table, again)
    assert again.calls == []  # every video is reused

    forced = FakeClient(prefer=("Law",))
    run(context, table, forced, force=True)
    assert len(forced.calls) == 2
    assert list(labels(context)[long]["pred_umd_categories"])[0] == "Law"
    assert "\"k\"" not in (context.umd_dir / "umd_identity.json").read_text(encoding="utf-8")


def test_low_confidence_warning_follows_the_threshold(tmp_path, table, monkeypatch):
    _, long = videos(tmp_path, monkeypatch)
    run(Context(tmp_path), table, FakeClient(prefer=("Crime",), top=0.29))
    assert labels(Context(tmp_path))[long]["warning"] == umd.LOW_CONFIDENCE
    # A new threshold relabels the warning without asking the model again.
    client = FakeClient(prefer=("Crime",))
    run(Context(tmp_path, {"low_confidence": 0.2}), table, client)
    assert client.calls == [] and labels(Context(tmp_path))[long]["warning"] is None
    # Exactly at the threshold is not low.
    run(Context(tmp_path, {"low_confidence": 0.29}), table, client)
    assert labels(Context(tmp_path))[long]["warning"] is None


def test_failed_videos_are_retried(tmp_path, table, monkeypatch):
    short, long = videos(tmp_path, monkeypatch)
    context = Context(tmp_path)
    assert run(context, table, FakeClient(prefer=("Soccer",), fail=True))["failure_count"] == 2
    row = labels(context)[short]
    assert "boom" in row["error"] and row["pred_umd_categories"] is None
    assert row["warning"] is None
    retry = FakeClient(prefer=("Soccer",))
    assert run(context, table, retry, content_ids=[long])["failure_count"] == 0
    assert len(retry.calls) == 1
    rows = labels(context)
    assert "error" not in rows[long] and "error" in rows[short]  # other videos are kept


def test_changed_table_or_frame_limit_discards_old_labels(tmp_path, table, monkeypatch):
    videos(tmp_path, monkeypatch)
    run(Context(tmp_path), table, FakeClient(prefer=("Soccer",)))
    fewer = FakeClient(prefer=("Soccer",))
    run(Context(tmp_path, {"max_frames": 10}), table, fewer)
    assert sorted(call["images"] for call in fewer.calls) == [3, 10]
    table.write_text(TABLE + "Kids,1,Family\n", encoding="utf-8")
    changed = FakeClient(prefer=("Family",))
    run(Context(tmp_path, {"max_frames": 10}), table, changed)
    assert len(changed.calls) == 2


@pytest.mark.parametrize("iab_settings, umd_settings, message", [
    ({"api_key": "YOUR_GEMINI_API_KEY"}, {}, "api_key"),
    ({}, {"max_frames": 0}, "max_frames"),
    ({}, {"low_confidence": 2}, "low_confidence"),
])
def test_bad_settings_fail_before_any_request(
    tmp_path, table, iab_settings, umd_settings, message
):
    client = FakeClient(prefer=("Soccer",))
    with pytest.raises(ValueError, match=message):
        run(Context(tmp_path, umd_settings, iab_settings), table, client)
    assert client.calls == []


def test_cli_lists_the_step(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "classify-umd-categories" in capsys.readouterr().out
