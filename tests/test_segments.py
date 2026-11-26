"""Segment labelling: table parsing, option wording, the single question, resume and CLI flags."""

import json

import pytest

from artifact_io import read_jsonl
from extraction import cli, iab, segments
from pipeline_runtime import RunContext

WATCH = "Viewers in the top 20% of those who watch {} contents on linear TV, based on ..."
TABLE = (
    "Category \tSubcategory\tData Source\tSegment Name\tAudience Definition\n"
    f"Media\tGenre\tTV \tAction\t{WATCH.format('action or adventure')}\n"
    "Media\tGenre\tTV \tCrime Drama\t\n"
    f"Media\tGenre\tTV \tTalk Show\t{WATCH.format('talkshow')}\n"
    f"Media\tSports\tTV \tWater Sports\t{WATCH.format('water sports')}"
)


class FakeClient:
    """Answers each question with the first option it was told to prefer."""

    def __init__(self, prefer, fail=()):
        self.prefer, self.fail, self.calls = prefer, set(fail), []

    def choose(self, images, instructions, options):
        self.calls.append(tuple(options))
        if self.fail and len(images) in self.fail:
            raise RuntimeError("boom")
        choice = next(key for key in self.prefer if key in options)
        probabilities = {key: (0.9 if key == choice else 0.1 / (len(options) - 1))
                         for key in options}
        return choice, 0.9, probabilities


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "segments.csv"
    path.write_text(TABLE, encoding="utf-8")
    return path


def test_load_segments_words_each_option(table):
    assert segments.load_segments(table) == {
        "action": ("Action", "Action (action or adventure)"),
        "crime_drama": ("Crime Drama", "Crime Drama"),
        "talk_show": ("Talk Show", "Talk Show"),
        "water_sports": ("Water Sports", "Water Sports"),
    }


@pytest.mark.parametrize("text, message", [
    ("Name\tDefinition\nAction\t\n", "header"),
    (TABLE + "\nMedia\tGenre\tTV \taction\t\n", "duplicate"),
    (TABLE + "\nMedia\tGenre\tTV \tNone\t\n", "reserved"),
    (TABLE.split("\n")[0] + "\nMedia\tGenre\tTV \tAction\t\n", "fewer than two"),
])
def test_load_segments_rejects_bad_tables(tmp_path, text, message):
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        segments.load_segments(path)


def test_repository_table_has_31_segments():
    path = RunContext.load().root / "audience_collectives_segment_repository.csv"
    if not path.is_file():
        pytest.skip("segment table not present")
    table = segments.load_segments(path)
    assert len(table) == 31
    assert table["kids"] == ("Kids", "Kids (children)")
    assert table["reality_cooking"] == ("Reality: Cooking", "Reality: Cooking")


def test_classify_asks_one_question_over_every_segment(table):
    client = FakeClient(prefer=("crime_drama",))
    label = segments.classify_scene(client, [], segments.load_segments(table))
    assert client.calls == [("action", "crime_drama", "talk_show", "water_sports", "none")]
    assert label["segment_id"] == "crime_drama" and label["segment"] == "Crime Drama"
    assert label["confidence"] == 0.9
    assert list(label["probabilities"])[0] == "crime_drama" and len(label["probabilities"]) == 5


def test_classify_keeps_no_segment_when_none_fits(table):
    label = segments.classify_scene(
        FakeClient(prefer=("none",)), [], segments.load_segments(table)
    )
    assert label["segment_id"] is None and label["segment"] is None
    assert label["confidence"] == 0.9


class Context:
    def __init__(self, root, settings):
        self.root, self.run_id = root, "T"
        self.config = {"extraction": {"iab": settings}}
        self.segment_dir = root / "segment_gt"


def visuals(tmp_path, monkeypatch, scenes=2):
    frames = tmp_path / "frames"
    frames.mkdir()
    windows = [{"keyframe_timestamps": [index + 0.5] * (index + 1)} for index in range(scenes)]
    for window in windows:
        for timestamp in window["keyframe_timestamps"]:
            (frames / f"{iab.timestamp_stem(timestamp)}.png").write_bytes(b"png")
    stamps = tmp_path / "timestamps.json"
    stamps.write_text(json.dumps(windows), encoding="utf-8")
    row = {"content_id": "vid", "frames_dir": str(frames), "timestamp_json": str(stamps)}
    monkeypatch.setattr(iab, "visual_rows", lambda context, split=None, limit=None: [row])
    monkeypatch.setattr(segments, "log_step_start", lambda *args, **kwargs: None)


def run(context, table, client, **options):
    return segments.extract_segment_gt(
        context, segments=str(table), client_factory=lambda **_: client, **options
    )


def test_step_writes_labels_and_resumes(tmp_path, table, monkeypatch):
    visuals(tmp_path, monkeypatch)
    context = Context(tmp_path, {"api_key": "k", "threads": 2})
    # The second scene has two keyframes; fail it on the first run only.
    first = FakeClient(prefer=("action",), fail={2})
    assert run(context, table, first)["failure_count"] == 1
    rows = read_jsonl(context.segment_dir / "vid.jsonl")
    assert rows[0]["segment"] == "Action" and "error" in rows[1]

    second = FakeClient(prefer=("none",))
    assert run(context, table, second)["failure_count"] == 0
    assert len(second.calls) == 1  # only the failed scene is asked again
    rows = read_jsonl(context.segment_dir / "vid.jsonl")
    assert [row["segment"] for row in rows] == ["Action", None]

    forced = FakeClient(prefer=("talk_show",))
    run(context, table, forced, force=True)
    assert len(forced.calls) == 2
    identity = (context.segment_dir / "segment_identity.json").read_text(encoding="utf-8")
    assert "segment-decision" in identity and '"k"' not in identity


def test_changed_table_discards_old_labels(tmp_path, table, monkeypatch):
    visuals(tmp_path, monkeypatch)
    context = Context(tmp_path, {"api_key": "k"})
    run(context, table, FakeClient(prefer=("action",)))
    table.write_text(TABLE + "\nMedia\tGenre\tTV \tPet\t\n", encoding="utf-8")
    again = FakeClient(prefer=("pet",))
    run(context, table, again)
    assert len(again.calls) == 2


def test_missing_table_names_the_flag(tmp_path):
    with pytest.raises(ValueError, match="--segments file not found"):
        run(Context(tmp_path, {"api_key": "k"}), tmp_path / "absent.csv", FakeClient(prefer=()))


@pytest.mark.parametrize("argv, message", [
    (["extract-segment-gt", "--run-id", "T"], "requires --segments"),
    (["extract-segment-gt", "--run-id", "T", "--segments", "s.csv", "--taxonomy", "t.csv"],
     "does not take --taxonomy"),
    (["extract-iab-gt", "--run-id", "T", "--taxonomy", "t.csv", "--segments", "s.csv"],
     "does not take --segments"),
    (["extract-graph-scenes", "--run-id", "T", "--schema", "p.md", "--model", "gemini",
      "--segments", "s.csv"], "does not take --segments"),
])
def test_cli_checks_flags_per_step(argv, message, capsys):
    assert cli.main(argv) == 1
    assert message in capsys.readouterr().err
