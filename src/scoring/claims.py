"""Split one scene graph into atomic claims a decision model can check.

Units follow the fields that are embedded downstream: Context (medium, format, topics),
Entity (one claim per entity, its attributes included) and Action (one claim per action,
its target, tool and place included). The judge reads every keyframe again for each claim,
so attributes and roles are not claims of their own. An action's parent is its actor.
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
UNITS = ("context", "entity", "action")


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
        claims.append({
            "id": f"entity:{entity['id']}", "unit": "entity", "kind": "entity",
            "value": entity["name"], "attributes": list(entity["attributes"]),
        })
    for index, action in enumerate(graph["actions"]):
        actor = entities.get(action["actor"])
        # A role the graph leaves unknown or states as none adds nothing to check.
        roles = {role: describe(entities[action[role]]) for role in ROLES
                 if action[role] not in (None, "unknown")}
        claims.append({
            "id": f"action:{index}", "unit": "action", "kind": "action", "value": action["action"],
            "actor": describe(actor) if actor else None, **roles,
            **({"parent": f"entity:{actor['id']}"} if actor else {}),
        })
    return claims


def foil_claims(graph: dict, pool: dict, *, seed: str, entities: int = 1, actions: int = 0):
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
        {"id": f"foil:entity:{index}", "unit": "foil", "kind": "entity", "value": name,
         "attributes": []}
        for index, name in enumerate(rng.sample(names, min(entities, len(names))))
    ]
    actor = next((describe(entity) for entity in graph["entities"]), None)
    foils += [
        {"id": f"foil:action:{index}", "unit": "foil", "kind": "action", "value": phrase,
         "actor": actor}
        for index, phrase in enumerate(rng.sample(phrases, min(actions, len(phrases))))
    ]
    return foils
