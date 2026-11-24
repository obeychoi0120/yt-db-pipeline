import json

import numpy as np
import pytest

from evaluation.agreement import compare_graphs, empty_agreement, match_items
from extraction.semantic_graph import parse_or_repair_graph
from extraction.semantic_graph.serialize import prompt_form, serialize_graph
from finetuning.dataset import build_record, scene_score
from finetuning.merge import find_best_checkpoint
from scoring.claims import foil_claims, graph_claims
from scoring.questions import claim_score, question, unit_scores
from scoring.report import bootstrap_interval
from tests.test_parser import PROMPTS, prompt_examples

ID_GRAPH = parse_or_repair_graph(
    "[Context]\nmedium: live_action\nformat: demonstration\ntopics: salmon sashimi; knife work\n"
    "[Entities]\nactor_1: woman; striped apron\ntarget_1: salmon fillet; raw\ntool_1: long knife\n"
    "[Actions]\nactor_1 - slicing thinly - target_1; tool_1; unknown\n"
    "actor_1 - plating - target_1; none; unknown"
).graph


@pytest.mark.parametrize("name,form", [("scene_graph_v8_2.md", "id"), ("scene_graph_v9_base.md", "name")])
def test_serialized_examples_parse_back_to_the_same_graph(name, form):
    assert prompt_form((PROMPTS / name).read_text(encoding="utf-8")) == form
    for text in prompt_examples(name):
        graph = parse_or_repair_graph(text).graph
        again = parse_or_repair_graph(serialize_graph(graph, form))
        assert again.parse_mode == "native" and again.graph == graph


def test_name_form_rejects_duplicate_names():
    graph = json.loads(json.dumps(ID_GRAPH))
    graph["entities"][1]["name"] = "woman"
    with pytest.raises(ValueError):
        serialize_graph(graph, "name")


def test_claims_fold_attributes_and_roles_into_their_parent():
    claims = {claim["id"]: claim for claim in graph_claims(ID_GRAPH)}
    assert set(claims) == {"medium", "format", "topic:0", "topic:1", "entity:actor_1",
                           "entity:target_1", "entity:tool_1", "action:0", "action:1"}
    assert {claim["unit"] for claim in claims.values()} == {"context", "entity", "action"}
    assert claims["entity:actor_1"]["attributes"] == ["striped apron"]
    assert claims["action:0"]["parent"] == "entity:actor_1"
    assert claims["action:0"]["target"] == "salmon fillet (raw)"
    assert claims["action:0"]["tool"] == "long knife"
    assert "place" not in claims["action:0"]  # unknown is not checked
    assert "tool" not in claims["action:1"]  # neither is an explicit none


def test_questions_and_scores():
    by_id = {claim["id"]: claim for claim in graph_claims(ID_GRAPH)}
    assert question(by_id["entity:actor_1"])["instructions"] == (
        "Is this visible in at least one keyframe, matching the whole description: "
        "woman (striped apron)?")
    assert question(by_id["entity:tool_1"])["instructions"] == (
        "Is this visible in at least one keyframe: long knife?")
    assert question(by_id["action:0"])["instructions"] == (
        "Do the keyframes show this action with every listed detail correct?\n"
        "Actor: woman (striped apron)\nAction: \"slicing thinly\"\n"
        "Acted on: salmon fillet (raw)\nTool or means: long knife")
    assert question(by_id["action:1"])["instructions"].endswith(
        "Action: \"plating\"\nActed on: salmon fillet (raw)")
    assert set(question(by_id["action:0"])["criteria"]) == {
        "supported", "contradicted", "cannot_tell"}
    assert question({"kind": "format", "value": "news"}) is None
    assert claim_score({"kind": "format", "value": "news"}, None)["p"] == 0.0
    verdict = {"type": "choice", "choice": "supported",
               "probabilities": {"supported": 0.8, "contradicted": 0.1, "cannot_tell": 0.1}}
    assert claim_score(by_id["entity:actor_1"], verdict) == {"p": 0.8, "abstain": 0.1}
    assert claim_score(by_id["topic:0"], {"type": "score", "score": 1.0})["p"] == 0.5
    picked = {"type": "choice", "choice": "animation",
              "probabilities": {"live_action": 0.3, "animation": 0.7}}
    assert claim_score(by_id["medium"], picked) == {"p": 0.3, "judge_choice": "animation"}


def test_an_action_counts_no_more_than_its_actor():
    claims = [
        {"id": "entity:a", "unit": "entity", "p": 0.5},
        {"id": "action:0", "unit": "action", "p": 0.8, "parent": "entity:a"},
    ]
    scores = unit_scores(claims)
    assert scores["entity"]["score"] == pytest.approx(0.5)
    assert scores["action"]["score"] == pytest.approx(0.4)


def test_foils_come_from_other_videos_only():
    pool = {"entities": {"woman", "elephant", "tractor"}, "actions": {"slicing thinly", "plowing"}}
    foils = foil_claims(ID_GRAPH, pool, seed="1:video:0", entities=2, actions=1)
    assert {f["value"] for f in foils if f["kind"] == "entity"} == {"elephant", "tractor"}
    assert [f["value"] for f in foils if f["kind"] == "action"] == ["plowing"]
    assert foils == foil_claims(ID_GRAPH, pool, seed="1:video:0", entities=2, actions=1)


def test_agreement_is_one_for_identical_graphs_and_zero_when_missing():
    vocabulary = {}

    def encode(texts):
        for text in texts:
            vocabulary.setdefault(text, len(vocabulary))
        vectors = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, vocabulary[text]] = 1.0
        return vectors

    same = compare_graphs(ID_GRAPH, ID_GRAPH, encode, threshold=0.8)
    assert same["medium"] == same["format"] == 1.0
    assert all(same[key]["f1"] == 1.0 for key in ("topics", "entities", "actions"))
    assert same["roles"] == {"agreement": 1.0, "links": 8}
    other = json.loads(json.dumps(ID_GRAPH))
    other["actions"][0]["tool"] = None
    other["entities"][1]["name"] = "tuna steak"
    changed = compare_graphs(other, ID_GRAPH, encode, threshold=0.8)
    assert changed["entities"]["matched"] == 2
    assert changed["roles"]["agreement"] < 1.0
    assert empty_agreement(ID_GRAPH)["entities"]["recall"] == 0.0


def test_match_items_is_one_to_one():
    similarity = np.array([[0.9, 0.85], [0.95, 0.1]])
    assert sorted(match_items(similarity, 0.8)) == [(0, 1), (1, 0)]
    assert match_items(similarity, 0.99) == []


def test_sft_record_and_scene_score():
    record = build_record("PROMPT", "TARGET", ["a.png", "b.png"], "video", 3)
    assert record["messages"][0]["content"] == "<image><image>PROMPT"
    assert record["messages"][1] == {"role": "assistant", "content": "TARGET"}
    row = {"scores": {"context": {"score": 0.9}, "entity": {"score": 0.5}}}
    assert scene_score(row, ["context", "entity", "action"]) == pytest.approx(0.7)


def test_best_checkpoint_has_the_lowest_eval_loss(tmp_path):
    run = tmp_path / "v0-20260101-000000"
    for step in (10, 20, 30):
        (run / f"checkpoint-{step}").mkdir(parents=True)
    history = [{"step": 10, "eval_loss": 0.5}, {"step": 20, "eval_loss": 0.2},
               {"step": 30, "eval_loss": 0.3}]
    (run / "checkpoint-30" / "trainer_state.json").write_text(json.dumps({"log_history": history}))
    checkpoint, loss = find_best_checkpoint(tmp_path)
    assert checkpoint.name == "checkpoint-20" and loss == 0.2


def test_bootstrap_interval_contains_the_mean():
    interval = bootstrap_interval([0.2, 0.4, 0.6, 0.8])
    assert interval["low"] <= interval["mean"] <= interval["high"] and interval["videos"] == 4
    assert bootstrap_interval([0.5])["low"] is None


def test_saved_scores_match_only_an_unchanged_graph():
    from scoring.steps import _claim_keys

    claim = {"id": "entity:e1", "unit": "entity", "kind": "entity", "value": "a dog",
             "attributes": ["brown"]}
    saved = [{**claim, "p": 0.9, "effective": 0.9}]
    assert _claim_keys(saved) == _claim_keys([claim])
    assert _claim_keys(saved) != _claim_keys([{**claim, "attributes": ["white"]}])


def test_changed_settings_remove_old_scores(tmp_path):
    import json

    from scoring.steps import _drop_stale_scores

    identity = {"version": 1, "judge": "d3", "state": "s", "foils": {"foil_seed": 42}}
    (tmp_path / "scoring_identity.json").write_text(json.dumps({**identity, "backend": "x"}))
    (tmp_path / "a.jsonl").write_text("{}\n")
    _drop_stale_scores(tmp_path, identity)
    assert (tmp_path / "a.jsonl").exists()

    _drop_stale_scores(tmp_path, {**identity, "foils": {"foil_seed": 7}})
    assert not (tmp_path / "a.jsonl").exists()
    assert not (tmp_path / "scoring_identity.json").exists()


def test_video_line_averages_unit_scores_over_scenes():
    from scoring.steps import video_line

    rows = [
        {"scores": {"context": {"score": 0.8}, "entity": {"score": 0.6}}, "foils": [{"p": 0.2}]},
        {"scores": {"context": {"score": 0.6}, "entity": {"score": 1.0},
                    "action": {"score": 0.5}}, "foils": [{"p": None}]},
    ]
    assert video_line("vid", rows, 3) == (
        "[SCORE] vid: scenes=2/3 context=0.700 entity=0.800 action=0.500 foil=0.200")
