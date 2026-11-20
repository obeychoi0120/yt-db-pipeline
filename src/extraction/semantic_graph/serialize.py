"""Write a parsed graph back as prompt-format text (the LoRA target and d3 text input)."""

from __future__ import annotations

import re

from extraction.semantic_graph.parser import GraphTextError, _parse_context_actions

FORMS = ("id", "name")


def prompt_form(prompt_text: str) -> str:
    """"id" when the prompt's first example declares ``id: kind`` entities, else "name"."""
    examples = prompt_text.split("[Examples]", 1)[-1]
    for block in examples.split("[Context]")[1:]:
        lines = ("[Context]" + block).rstrip().splitlines()
        if "[Actions]" not in lines:
            continue
        end = next(
            (i for i, line in enumerate(lines) if not line.strip() and i > lines.index("[Actions]")),
            len(lines),
        )
        try:
            graph, named = _parse_context_actions("\n".join(lines[:end]), repair=False)
        except GraphTextError:
            continue
        if graph["entities"]:
            return "name" if named else "id"
    raise ValueError("the prompt has no example with entities to tell its output form")


def _reference(value, names=None):
    if value is None:
        return "none"
    if value == "unknown":
        return "unknown"
    return names[value] if names is not None else value


def serialize_graph(graph: dict, form: str = "id") -> str:
    """Canonical text of one graph in ID form (v8_2) or name form (v9_base)."""
    if form not in FORMS:
        raise ValueError(f"form must be one of {', '.join(FORMS)}")
    lines = [
        "[Context]",
        f"medium: {graph['medium']}",
        f"format: {graph['format']}",
        "topics: " + ("; ".join(graph["topics"]) if graph["topics"] else "none"),
        "[Entities]",
    ]
    names = None
    if form == "name":
        names = {entity["id"]: entity["name"] for entity in graph["entities"]}
        if len(set(names.values())) != len(names):
            raise ValueError("name form needs unique entity names")
        if any(re.search(r"[;:]| - ", name) for name in names.values()):
            raise ValueError("name form cannot hold names with delimiters")
    for entity in graph["entities"]:
        head = entity["name"] if form == "name" else f"{entity['id']}: {entity['name']}"
        lines.append("; ".join([head, *entity["attributes"]]))
    if not graph["entities"]:
        lines.append("none")
    lines.append("[Actions]")
    for action in graph["actions"]:
        body = " - ".join(
            [_reference(action["actor"], names), action["action"], _reference(action["target"], names)]
        )
        tool, place = _reference(action["tool"], names), _reference(action["place"], names)
        if form == "id":
            lines.append(f"{body}; {tool}; {place}")
        else:
            # Omitted slots mean tool none and place unknown.
            slots = ([f"tool: {tool}"] if tool != "none" else []) + (
                [f"place: {place}"] if place != "unknown" else []
            )
            lines.append("; ".join([body, *slots]))
    if not graph["actions"]:
        lines.append("none")
    return "\n".join(lines)
