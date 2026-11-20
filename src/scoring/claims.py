"""Split one scene graph into atomic claims a decision model can check.

Units follow the fields that are embedded downstream: Context (medium, format, topics),
Entity (existence, attributes), Action (occurrence) and Role (target, tool, place links).
A claim's parent is the claim it depends on: an attribute needs its entity, an action its
actor, a role link its action.
"""

from __future__ import annotations

import random

MEDIA = {
    "live_action": "Real people, objects or places filmed with a camera",
    "animation": "Animated, drawn or computer-generated imagery",
    "gameplay": "Footage of a video game being played",
    "screen_recording": "A recording of a computer or phone screen that is not a game",
    "mixed": "A mix of several of these media",
    "unknown": "Cannot be determined",
}
FORMATS = {
    "demonstration": "Showing how something is done or made",
    "explanation": "Explaining or presenting information",
    "review": "Evaluating or comparing a product, place or work",
    "narrative": "Telling a story or reporting events",
    "highlights": "A compilation of notable moments",
    "performance": "A performance such as music, dance, sport or eating",
    "interview": "An interview or conversation",
    "other": "Something else",
    "unknown": "Cannot be determined",
}
ROLES = ("target", "tool", "place")
UNITS = ("context", "entity", "action", "role")


def describe(entity: dict) -> str:
    """An entity as it is named in questions: kind plus its distinguishing attributes."""
    if entity["attributes"]:
        return f"{entity['name']} ({'; '.join(entity['attributes'])})"
    return entity["name"]


def graph_claims(graph: dict) -> list[dict]:
    entities = {entity["id"]: entity for entity in graph["entities"]}
    claims = [
        {"id": "medium", "unit": "context", "kind": "medium", "value": graph["medium"]},
        {"id": "format", "unit": "context", "kind": "format", "value": graph["format"]},
    ]
    for index, topic in enumerate(graph["topics"]):
        claims.append({"id": f"topic:{index}", "unit": "context", "kind": "topic", "value": topic})
    for entity in graph["entities"]:
        entity_id = f"entity:{entity['id']}"
        claims.append({"id": entity_id, "unit": "entity", "kind": "entity", "value": entity["name"]})
        for index, attribute in enumerate(entity["attributes"]):
            claims.append({
                "id": f"attribute:{entity['id']}:{index}", "unit": "entity", "kind": "attribute",
                "value": attribute, "entity": entity["name"], "parent": entity_id,
            })
    for index, action in enumerate(graph["actions"]):
        actor = entities.get(action["actor"])
        action_id = f"action:{index}"
        claims.append({
            "id": action_id, "unit": "action", "kind": "action", "value": action["action"],
            "actor": describe(actor) if actor else None,
            **({"parent": f"entity:{actor['id']}"} if actor else {}),
        })
        for role in ROLES:
            value = action[role]
            if value == "unknown":
                continue  # the graph makes no claim
            claims.append({
                "id": f"role:{index}:{role}", "unit": "role", "kind": role,
                "value": "none" if value is None else value,
                "value_name": "none" if value is None else describe(entities[value]),
                "action": action["action"], "actor": describe(actor) if actor else None,
                "options": {entity["id"]: describe(entity) for entity in graph["entities"]},
                "parent": action_id,
            })
    return claims


def foil_claims(graph: dict, pool: dict, *, seed: str, entities: int = 2, actions: int = 1):
    """Claims taken from other videos; a reliable judge rejects them.

    ``pool`` holds entity names and action phrases of the other videos. Names and phrases
    already in this graph are never used.
    """
    rng = random.Random(seed)
    own_names = {entity["name"].casefold() for entity in graph["entities"]}
    own_actions = {action["action"].casefold() for action in graph["actions"]}
    names = sorted(name for name in pool["entities"] if name.casefold() not in own_names)
    phrases = sorted(phrase for phrase in pool["actions"] if phrase.casefold() not in own_actions)
    foils = [
        {"id": f"foil:entity:{index}", "unit": "foil", "kind": "entity", "value": name}
        for index, name in enumerate(rng.sample(names, min(entities, len(names))))
    ]
    actor = next((describe(entity) for entity in graph["entities"]), None)
    foils += [
        {"id": f"foil:action:{index}", "unit": "foil", "kind": "action", "value": phrase,
         "actor": actor}
        for index, phrase in enumerate(rng.sample(phrases, min(actions, len(phrases))))
    ]
    return foils
