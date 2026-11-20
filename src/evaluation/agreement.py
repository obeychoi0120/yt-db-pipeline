"""Teacher agreement of one scene graph pair, on the embedded fields.

Topics, entities and actions are free text, so items are paired by embedding similarity:
the most similar pairs first, each item used once, pairs below the threshold left unmatched.
Role links are compared on the paired actions: a link agrees when both graphs leave it
empty in the same way or name entities that are paired with each other.
"""

from __future__ import annotations

import numpy as np

ROLES = ("actor", "target", "tool", "place")


def entity_text(entity: dict) -> str:
    """The text the downstream encoder embeds for one entity."""
    return f"Name: {entity['name']}\nAttributes: {', '.join(sorted(entity['attributes']))}"


def match_items(similarity: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Greedy one-to-one pairs (row, column) with similarity >= threshold, best first."""
    pairs, used_rows, used_columns = [], set(), set()
    if similarity.size == 0:
        return pairs
    order = np.dstack(np.unravel_index(np.argsort(-similarity, axis=None), similarity.shape))[0]
    for row, column in order:
        if similarity[row, column] < threshold:
            break
        if row not in used_rows and column not in used_columns:
            pairs.append((int(row), int(column)))
            used_rows.add(row)
            used_columns.add(column)
    return pairs


def _prf(matched: int, student: int, teacher: int) -> dict:
    precision = matched / student if student else (1.0 if not teacher else 0.0)
    recall = matched / teacher if teacher else (1.0 if not student else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "matched": matched, "student": student, "teacher": teacher}


def _pairs(encode, student_texts, teacher_texts, threshold):
    if not student_texts or not teacher_texts:
        return []
    similarity = encode(student_texts) @ encode(teacher_texts).T
    return match_items(similarity, threshold)


def compare_graphs(student: dict, teacher: dict, encode, *, threshold: float) -> dict:
    """Agreement of one student graph with the teacher graph of the same scene."""
    result = {
        "medium": float(student["medium"] == teacher["medium"]),
        "format": float(student["format"] == teacher["format"]),
    }
    topics = _pairs(encode, student["topics"], teacher["topics"], threshold)
    result["topics"] = _prf(len(topics), len(student["topics"]), len(teacher["topics"]))
    entities = _pairs(
        encode, [entity_text(e) for e in student["entities"]],
        [entity_text(e) for e in teacher["entities"]], threshold,
    )
    result["entities"] = _prf(len(entities), len(student["entities"]), len(teacher["entities"]))
    actions = _pairs(
        encode, [a["action"] for a in student["actions"]],
        [a["action"] for a in teacher["actions"]], threshold,
    )
    result["actions"] = _prf(len(actions), len(student["actions"]), len(teacher["actions"]))
    paired = {
        student["entities"][row]["id"]: teacher["entities"][column]["id"]
        for row, column in entities
    }
    agree = total = 0
    for row, column in actions:
        for role in ROLES:
            mine, theirs = student["actions"][row][role], teacher["actions"][column][role]
            total += 1
            if mine in (None, "unknown") or theirs in (None, "unknown"):
                agree += mine == theirs
            else:
                agree += paired.get(mine) == theirs
    result["roles"] = {"agreement": agree / total if total else None, "links": total}
    return result


def empty_agreement(teacher: dict) -> dict:
    """A scene the student has no usable graph for: nothing of the teacher is recovered."""
    zero = lambda count: _prf(0, 0, count) if count else _prf(0, 0, 0)  # noqa: E731
    return {
        "medium": 0.0, "format": 0.0,
        "topics": zero(len(teacher["topics"])), "entities": zero(len(teacher["entities"])),
        "actions": zero(len(teacher["actions"])), "roles": {"agreement": None, "links": 0},
        "missing": True,
    }
