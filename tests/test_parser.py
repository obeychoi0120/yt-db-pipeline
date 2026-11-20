"""The supported prompts' own examples must parse without repair."""

from pathlib import Path

import pytest

from extraction.semantic_graph import parse_or_repair_graph

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def prompt_examples(name):
    text = (PROMPTS / name).read_text(encoding="utf-8")
    examples = text.split("[Examples]", 1)[1]
    blocks = examples.split("[Context]")[1:]
    graphs = []
    for block in blocks:
        lines = ("[Context]" + block).rstrip().splitlines()
        # The line after a graph's last action introduces the next example.
        end = len(lines)
        for index, line in enumerate(lines):
            if line.strip() == "" and index > lines.index("[Actions]"):
                end = index
                break
        graphs.append("\n".join(lines[:end]))
    return graphs


@pytest.mark.parametrize("name", ["scene_graph_v8_2.md", "scene_graph_v9_base.md"])
def test_prompt_examples_parse_natively(name):
    graphs = prompt_examples(name)
    assert len(graphs) == 3
    for text in graphs:
        parsed = parse_or_repair_graph(text)
        assert parsed.error is None, (text, parsed.error)
        assert parsed.parse_mode == "native", text
        assert set(parsed.graph) == {"medium", "format", "topics", "entities", "actions"}


def test_id_form_roles_reference_declared_ids():
    parsed = parse_or_repair_graph(
        "[Context]\nmedium: live_action\nformat: demonstration\ntopics: salmon sashimi\n"
        "[Entities]\nactor_1: woman; striped apron\ntarget_1: salmon fillet\ntool_1: knife\n"
        "[Actions]\nactor_1 - slicing thinly - target_1; tool_1; unknown"
    )
    assert parsed.graph["actions"] == [
        {"actor": "actor_1", "action": "slicing thinly", "target": "target_1",
         "tool": "tool_1", "place": "unknown"}
    ]
    assert parsed.graph["entities"][0] == {
        "id": "actor_1", "name": "woman", "attributes": ["striped apron"]
    }


def test_name_form_labelled_slots_take_defaults():
    parsed = parse_or_repair_graph(
        "[Context]\nmedium: live_action\nformat: demonstration\ntopics: woodworking\n"
        "[Entities]\nman1; wearing gloves\nwooden board\nsanding block\n"
        "[Actions]\nman1 - sanding - wooden board; tool: sanding block"
    )
    assert parsed.parse_mode == "native"
    names = {entity["id"]: entity["name"] for entity in parsed.graph["entities"]}
    action = parsed.graph["actions"][0]
    assert (names[action["actor"]], names[action["target"]], names[action["tool"]]) == (
        "man1", "wooden board", "sanding block"
    )
    assert action["place"] == "unknown"


def test_broken_output_fails_instead_of_guessing():
    parsed = parse_or_repair_graph('{"medium": "live_action"}')
    assert parsed.graph is None and parsed.warning == ("PARSE_FAILED",)
