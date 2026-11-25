"""IAB labelling: taxonomy parsing, tier-by-tier descent, resume and CLI validation."""

import json

import pytest

from artifact_io import read_jsonl
from extraction import cli, iab
from pipeline_runtime import RunContext

TABLE = (
    "Relational ID System\t\t\tTiered Categories\n"
    "Unique ID\tParent\tName\tTier 1\tTier 2\tTier 3\n"
    "1\t\tSports\tSports\n"
    "2\t1\tCycling\tSports\tCycling\n"
    "3\t1\tSoccer\tSports\tSoccer\n"
    "4\t2\tRoad Cycling\tSports\tCycling\tRoad Cycling\n"
    "5\t\tTravel\tTravel\n"
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
    path = tmp_path / "taxonomy.csv"
    path.write_text(TABLE, encoding="utf-8")
    return path


def test_load_taxonomy_builds_the_tree(table):
    tree = iab.load_taxonomy(table)
    assert tree.children[None] == ("1", "5")
    assert tree.children["1"] == ("2", "3")
    assert tree.path("4") == ["Sports", "Cycling", "Road Cycling"]


@pytest.mark.parametrize("text, message", [
    ("a,b,c\n1,,Sports\n", "header"),
    (TABLE + "6\t99\tOrphan\n", "missing parent"),
    (TABLE + "5\t\tAgain\n", "duplicate"),
])
def test_load_taxonomy_rejects_bad_tables(tmp_path, text, message):
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        iab.load_taxonomy(path)


def test_repository_taxonomy_fits_the_option_limit():
    path = RunContext.load().root / "content_taxonomy_3.1_converted.csv"
    if not path.is_file():
        pytest.skip("taxonomy table not present")
    tree = iab.load_taxonomy(path)
    assert len(tree.children[None]) == 37
    assert max(len(ids) for ids in tree.children.values()) < iab.MAX_OPTIONS


def test_classify_descends_to_a_leaf(table):
    client = FakeClient(prefer=("1", "2", "4"))
    label = iab.classify_scene(client, [], iab.load_taxonomy(table))
    assert label["iab_id"] == "4"
    assert label["iab_path"] == ["Sports", "Cycling", "Road Cycling"]
    assert [tier["tier"] for tier in label["tiers"]] == [1, 2, 3]
    assert "stop_confidence" not in label
    # Tier 1 has no "none"; every lower question offers it.
    assert client.calls == [("1", "5"), ("2", "3", "none"), ("4", "none")]


def test_classify_stops_when_no_child_fits(table):
    client = FakeClient(prefer=("1", "none"))
    label = iab.classify_scene(client, [], iab.load_taxonomy(table))
    assert label["iab_id"] == "1" and label["iab_path"] == ["Sports"]
    assert label["stop_confidence"] == 0.9
    assert len(label["tiers"]) == 1


def test_classify_stops_at_a_childless_top_category(table):
    client = FakeClient(prefer=("5",))
    label = iab.classify_scene(client, [], iab.load_taxonomy(table))
    assert label["iab_id"] == "5" and len(client.calls) == 1


class Context:
    def __init__(self, root, settings):
        self.root, self.run_id = root, "T"
        self.config = {"extraction": {"iab": settings}}
        self.iab_dir = root / "iab_gt"


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
    monkeypatch.setattr(iab, "log_step_start", lambda *args, **kwargs: None)


def run(context, table, client, **options):
    return iab.extract_iab_gt(
        context, taxonomy=str(table), client_factory=lambda **_: client, **options
    )


def test_step_writes_labels_and_resumes(tmp_path, table, monkeypatch):
    visuals(tmp_path, monkeypatch)
    context = Context(tmp_path, {"api_key": "k", "threads": 2})
    # The second scene has two keyframes; fail it on the first run only.
    first = FakeClient(prefer=("1", "none"), fail={2})
    assert run(context, table, first)["failure_count"] == 1
    rows = read_jsonl(context.iab_dir / "vid.jsonl")
    assert [row["scene_idx"] for row in rows] == [0, 1]
    assert rows[0]["iab_name"] == "Sports" and "error" in rows[1]

    second = FakeClient(prefer=("5",))
    assert run(context, table, second)["failure_count"] == 0
    assert len(second.calls) == 1  # only the failed scene is asked again
    rows = read_jsonl(context.iab_dir / "vid.jsonl")
    assert [row["iab_name"] for row in rows] == ["Sports", "Travel"]

    forced = FakeClient(prefer=("5",))
    run(context, table, forced, force=True)
    assert len(forced.calls) == 2
    assert "k" not in (context.iab_dir / "iab_identity.json").read_text(encoding="utf-8")


def test_changed_taxonomy_discards_old_labels(tmp_path, table, monkeypatch):
    visuals(tmp_path, monkeypatch)
    context = Context(tmp_path, {"api_key": "k"})
    run(context, table, FakeClient(prefer=("5",)))
    table.write_text(TABLE + "6\t\tPets\tPets\n", encoding="utf-8")
    again = FakeClient(prefer=("6",))
    run(context, table, again)
    assert len(again.calls) == 2


@pytest.mark.parametrize("settings", [{}, {"api_key": "YOUR_GEMINI_API_KEY"}, {"api_key": " "}])
def test_missing_api_key_fails_before_any_request(tmp_path, table, settings):
    with pytest.raises(ValueError, match="api_key"):
        run(Context(tmp_path, settings), table, FakeClient(prefer=("5",)))


@pytest.mark.parametrize("argv, message", [
    (["extract-iab-gt", "--run-id", "T"], "requires --taxonomy"),
    (["extract-iab-gt", "--run-id", "T", "--taxonomy", "t.csv", "--model", "gemini"],
     "does not take --model"),
    (["extract-graph-scenes", "--run-id", "T", "--model", "gemini"], "requires --schema"),
    (["extract-graph-scenes", "--run-id", "T", "--schema", "p.md", "--model", "gemini",
      "--taxonomy", "t.csv"], "does not take --taxonomy"),
])
def test_cli_checks_flags_per_step(argv, message, capsys):
    assert cli.main(argv) == 1
    assert message in capsys.readouterr().err
