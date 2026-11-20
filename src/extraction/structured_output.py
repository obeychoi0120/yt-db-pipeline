"""Scene graph contract, independent of the selected prompt filename."""

from __future__ import annotations

from copy import deepcopy

from extraction.graph_warnings import PARSE_FAILED


def _object(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(items):
    return {"type": "array", "items": items}


STRING = {"type": "string", "minLength": 1}
REFERENCE = {"type": ["string", "null"], "minLength": 1}
GRAPH_JSON_SCHEMA = _object(
    {
        "medium": STRING,
        "format": STRING,
        "topics": _array(STRING),
        "entities": _array(
            _object(
                {
                    "id": STRING,
                    "name": STRING,
                    "attributes": _array(STRING),
                }
            )
        ),
        "actions": _array(
            _object(
                {
                    "actor": REFERENCE,
                    "action": STRING,
                    "target": REFERENCE,
                    "tool": REFERENCE,
                    "place": REFERENCE,
                },
            )
        ),
    }
)


class OutputValidationError(ValueError):
    def __init__(self, message, tags=(PARSE_FAILED,)):
        super().__init__(message)
        self.tags = list(dict.fromkeys(tags))


def validate_graph_structure(value, path="graph"):
    """Require readable fields and types, without validating Graph semantics."""
    _validate_shape(value, GRAPH_JSON_SCHEMA, path)


def normalize_graph_references(graph):
    """Use unknown for undeclared role IDs, keeping all entities and actions."""
    repaired, changed = deepcopy(graph), False
    _validate_shape(repaired, GRAPH_JSON_SCHEMA, "graph")
    ids = {entity["id"] for entity in repaired["entities"]}
    for action in repaired["actions"]:
        for field in ("actor", "target", "tool", "place"):
            value = action[field]
            if value not in (None, "unknown") and value not in ids:
                action[field] = "unknown"
                changed = True
    return repaired, changed


def _validate_shape(value, schema, path):
    kind = schema["type"]
    if isinstance(kind, list):
        if value is None and "null" in kind:
            return
        kind = "string"
    expected = {"object": dict, "array": list, "string": str}[kind]
    if not isinstance(value, expected):
        raise OutputValidationError(f"{path}: expected {kind}")
    if kind == "object":
        if set(schema["required"]) - set(value):
            raise OutputValidationError(f"{path}: missing required fields")
        if set(value) - set(schema["properties"]):
            raise OutputValidationError(f"{path}: extra fields")
        for key, child in schema["properties"].items():
            if key in value:
                _validate_shape(value[key], child, f"{path}.{key}")
    elif kind == "array":
        for index, item in enumerate(value):
            _validate_shape(item, schema["items"], f"{path}[{index}]")
    elif not value.strip():
        raise OutputValidationError(f"{path}: empty string")
