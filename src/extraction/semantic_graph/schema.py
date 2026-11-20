"""Context, Entities and Actions graphs."""

from __future__ import annotations
from extraction.structured_output import OutputValidationError, validate_graph_structure

GRAPH_SCHEMA_VERSION = "scene-graph/v8"


def graph_semantic_warnings(graph):
    try:
        validate_graph_structure(graph)
    except OutputValidationError as exc:
        return exc.tags
    return []
